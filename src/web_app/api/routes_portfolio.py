"""Portfolio endpoints: read/replace a user's holdings and analyze them (deterministic; narration optional)."""
from __future__ import annotations

import asyncio
from typing import Callable

from fastapi import APIRouter, Depends, Query, Request

from src.agents.base import AgentRequest
from src.agents.portfolio import PortfolioAnalysis
from src.core.freshness import DataStatus, combine_status, humanize_age
from src.core.market_calc import DEFAULT_BENCHMARK, compute_trend, rebased, relative_performance
from src.core.models import Portfolio
from src.core.portfolio_history import interpret, value_series
from src.utils.validators import normalize_ticker
from src.web_app.errors import ApiError
from src.data.repositories import PortfolioRepository, UserRepository
from src.web_app.api.models import NOT_FOUND, TOO_MANY, PortfolioAnalysisOut, PortfolioComparisonOut, PortfolioIn, PortfolioOut, UserIdPath
from src.web_app.api.services import ApiServices
from src.web_app.ratelimit import client_key, enforce

MAX_COMPARED_HOLDINGS = 20
EXPLAIN_QUESTION = "Explain my portfolio analysis in plain language."


def _out(user_id: str, a: PortfolioAnalysis, *, what_if: bool, explanation: str | None = None) -> PortfolioAnalysisOut:
    if a.empty:
        return PortfolioAnalysisOut(user_id=user_id, empty=True, what_if=what_if,
                                    stated_risk_tolerance=a.stated_risk.value if a.stated_risk else None)
    return PortfolioAnalysisOut(
        user_id=user_id, what_if=what_if, metrics=a.metrics, holdings=a.rows, data_status=a.status,
        notice=" ".join(a.notices) or None, notices=a.notices, priced_with=a.priced_with, unpriced=a.unpriced,
        sample_quotes_ignored=a.sample_ignored, errors=a.errors,
        stated_risk_tolerance=a.stated_risk.value if a.stated_risk else None, explanation=explanation)


def _history_notices(non_live: dict[DataStatus, list[tuple[str, float | None]]]) -> list[str]:
    """One sentence per kind of non-live history (instead of one per holding), naming the oldest age."""
    phrases = {DataStatus.cached: "a saved copy rather than a fresh lookup", DataStatus.stale: "the last known values, which may be out of date"}
    out = []
    for status, phrase in phrases.items():
        items = non_live.get(status)
        if items:
            names = sorted(s for s, _ in items)
            ages = [a for _, a in items if a is not None]
            who = f"{', '.join(names[:-1])} and {names[-1]} are" if len(names) > 1 else f"{names[0]} is"
            out.append(f"Price history for {who} from {phrase}" + (f" (oldest about {humanize_age(max(ages))} old)." if ages else "."))
    return out


def build_portfolio_router(get_services: Callable[[], ApiServices]) -> APIRouter:
    router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

    @router.get("/{user_id}", response_model=PortfolioOut, responses=NOT_FOUND, summary="A user's saved holdings and cash")
    def get_portfolio(user_id: UserIdPath, request: Request, svc: ApiServices = Depends(get_services)) -> PortfolioOut:
        enforce(svc.limiters["read"], client_key(request))
        profile = svc.require_user(user_id)
        p = PortfolioRepository(svc.db).get(profile.user_id)
        return PortfolioOut(user_id=profile.user_id, holdings=p.holdings, cash=p.cash,
                            synthetic=UserRepository(svc.db).is_synthetic(profile.user_id))

    @router.put("/{user_id}", response_model=PortfolioOut, summary="Replace a user's holdings and cash",
                description="Atomically replaces the saved portfolio. A new user id is created on first save. "
                            f"Up to 100 holdings; each ticker once; tickers are normalized to upper case.")
    def put_portfolio(user_id: UserIdPath, body: PortfolioIn, request: Request, svc: ApiServices = Depends(get_services)) -> PortfolioOut:
        enforce(svc.limiters["read"], client_key(request))
        profile, _created = UserRepository(svc.db).ensure(user_id)
        PortfolioRepository(svc.db).replace(Portfolio(user_id=profile.user_id, holdings=body.holdings, cash=body.cash))
        return PortfolioOut(user_id=profile.user_id, holdings=body.holdings, cash=body.cash,
                            synthetic=UserRepository(svc.db).is_synthetic(profile.user_id))

    @router.get("/{user_id}/comparison", response_model=PortfolioComparisonOut, responses={**NOT_FOUND, **TOO_MANY},
                summary="How today's holdings would have performed against a benchmark",
                description="Replays the CURRENT holdings (and cash) over past prices and compares the result with a benchmark (default SPY, an S&P 500 "
                            "proxy), both rebased to 100. Hypothetical: ignores when you actually bought, taxes and fees. Holdings without usable history "
                            "are left out and listed with the coverage; SAMPLE history is never used. Descriptive, not a forecast or advice.")
    async def comparison(user_id: UserIdPath, request: Request, benchmark: str = Query(default=DEFAULT_BENCHMARK, max_length=12),
                         days: int = Query(default=180, ge=30, le=365), svc: ApiServices = Depends(get_services)) -> PortfolioComparisonOut:
        enforce(svc.limiters["read"], client_key(request))
        profile = svc.require_user(user_id)
        bench_sym = normalize_ticker(benchmark)
        if bench_sym is None:
            raise ApiError(422, "Not a valid ticker symbol", "invalid_symbol")
        saved = PortfolioRepository(svc.db).get(profile.user_id)
        holdings = [h for h in saved.holdings if h.shares > 0][:MAX_COMPARED_HOLDINGS]
        if not holdings:
            return PortfolioComparisonOut(user_id=profile.user_id, empty=True, days=days)

        async def history(symbol: str):
            try:
                _trend, hist = await asyncio.to_thread(svc.market.get_trend, symbol, days)
                return hist, None
            except Exception:
                return None, "Price history unavailable"

        tickers = list(dict.fromkeys(h.ticker for h in holdings))
        fetched = dict(zip([*tickers, bench_sym], await asyncio.gather(*(history(s) for s in [*tickers, bench_sym]))))
        excluded: dict[str, str] = {}
        histories, statuses, notices = {}, [], []
        non_live: dict[DataStatus, list[tuple[str, float | None]]] = {}
        for t in tickers:
            hist, error = fetched[t]
            if hist is None:
                excluded[t] = error or "Price history unavailable"
            elif hist.status is DataStatus.sample:
                excluded[t] = "Only SAMPLE data was available, which is never used for your portfolio"
            else:
                histories[t] = hist.points
                statuses.append(hist.status)
                if hist.status is not DataStatus.live:
                    non_live.setdefault(hist.status, []).append((t, hist.age_seconds))
        notices += _history_notices(non_live)
        series, left_out, coverage = value_series(holdings, histories, saved.cash)
        for t in left_out:
            excluded.setdefault(t, "Not enough shared price history")
        if not series:
            return PortfolioComparisonOut(user_id=profile.user_id, benchmark=None, excluded=excluded, days=days,
                                          notices=[*dict.fromkeys(notices), "There was not enough price history to compare your portfolio right now."],
                                          insights=[], data_status=combine_status(statuses) if statuses else DataStatus.live)
        port_trend = compute_trend("Your mix", series)
        bench_hist, _ = fetched[bench_sym]
        bench_trend, bench_series = None, None
        if bench_hist is not None and bench_hist.status is not DataStatus.sample:
            bench_trend, bench_series = compute_trend(bench_sym, bench_hist.points), bench_hist.points
            statuses.append(bench_hist.status)
            if bench_hist.status is not DataStatus.live:
                notices += _history_notices({bench_hist.status: [(bench_sym, bench_hist.age_seconds)]})
        else:
            notices.append(f"The comparison with {bench_sym} is unavailable right now"
                           + (" (only SAMPLE data was available)." if bench_hist is not None else "."))
        comparison = rebased({"Your mix": series, **({bench_sym: bench_series} if bench_series else {})})
        return PortfolioComparisonOut(
            user_id=profile.user_id, benchmark=bench_sym if bench_trend else None, included=sorted(histories), excluded=excluded,
            coverage_pct=coverage, days=days, portfolio_trend=port_trend, benchmark_trend=bench_trend,
            relative=relative_performance(port_trend, bench_trend) if bench_trend else {}, comparison=comparison,
            insights=interpret(port_trend, bench_trend, bench_sym if bench_trend else None, coverage_pct=coverage, excluded=sorted(excluded)),
            data_status=combine_status(statuses), notices=list(dict.fromkeys(notices)))

    @router.get("/{user_id}/analysis", response_model=PortfolioAnalysisOut, responses={**NOT_FOUND, **TOO_MANY},
                summary="Value, allocation, diversification, fees and risk of the saved portfolio",
                description="Deterministic (no LLM). Prices come from the market chain; cached / last-known prices are disclosed in "
                            "`notices`, and SAMPLE prices are never used for valuation. `explain=true` adds a plain-language narration.")
    async def analysis(user_id: UserIdPath, request: Request, explain: bool = Query(default=False),
                       svc: ApiServices = Depends(get_services)) -> PortfolioAnalysisOut:
        enforce(svc.limiters["expensive" if explain else "read"], client_key(request))
        profile = svc.require_user(user_id)
        a = await svc.portfolio_agent.analyze(profile.user_id)
        explanation = None
        if explain and not a.empty:
            result = await svc.portfolio_agent.arun(AgentRequest(question=EXPLAIN_QUESTION, user_id=profile.user_id))
            explanation = result.content if result.ok else None
        return _out(profile.user_id, a, what_if=False, explanation=explanation)

    @router.post("/{user_id}/analysis", response_model=PortfolioAnalysisOut, responses={**NOT_FOUND, **TOO_MANY},
                 summary="What-if: analyze a submitted portfolio without saving it",
                 description="Analyzes the holdings in the request body (e.g. a user typing a portfolio) using the same engine. Nothing is stored.")
    async def analyze_submitted(user_id: UserIdPath, body: PortfolioIn, request: Request, explain: bool = Query(default=False),
                                svc: ApiServices = Depends(get_services)) -> PortfolioAnalysisOut:
        enforce(svc.limiters["expensive" if explain else "read"], client_key(request))
        profile = svc.require_user(user_id)
        submitted = Portfolio(user_id=profile.user_id, holdings=body.holdings, cash=body.cash)
        a = await svc.portfolio_agent.analyze(profile.user_id, submitted)
        explanation = None
        if explain and not a.empty:
            result = await svc.portfolio_agent.arun(AgentRequest(question=EXPLAIN_QUESTION, user_id=profile.user_id,
                                                                 context={"portfolio": submitted}))
            explanation = result.content if result.ok else None
        return _out(profile.user_id, a, what_if=True, explanation=explanation)

    return router
