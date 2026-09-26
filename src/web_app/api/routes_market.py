"""Market endpoints: quotes, price-history trends, and provider health. Every non-live value carries a notice."""
from __future__ import annotations

import asyncio
from typing import Callable

from fastapi import APIRouter, Depends, Path, Query, Request

from src.core.freshness import DataStatus, combine_status
from src.core.market_calc import DEFAULT_BENCHMARK, SEGMENTS, interpret, rebased, relative_performance
from src.utils.validators import normalize_ticker
from src.web_app.api.models import TOO_MANY, ContextOut, OverviewOut, ProviderStatusOut, QuotesOut, SegmentOut, TrendOut
from src.web_app.api.services import ApiServices
from src.web_app.errors import ApiError
from src.web_app.ratelimit import client_key, enforce

MAX_SYMBOLS = 25


def build_market_router(get_services: Callable[[], ApiServices]) -> APIRouter:
    router = APIRouter(prefix="/api/market", tags=["market"])

    @router.get("/quotes", response_model=QuotesOut, responses=TOO_MANY, summary="Latest quotes for up to 25 symbols",
                description="Providers: Alpha Vantage, then yFinance, then the last saved value, then SAMPLE data (only when providers are "
                            "down). Each quote has `data_status` (live/cached/stale/sample) and a plain-language `notice` unless live. "
                            "Invalid or unknown symbols are reported in `errors`; the rest are still returned.")
    def quotes(request: Request, symbols: str = Query(min_length=1, max_length=400, description="Comma-separated, e.g. AAPL,MSFT,BTC-USD"),
               svc: ApiServices = Depends(get_services)) -> QuotesOut:
        enforce(svc.limiters["read"], client_key(request))
        requested = [s for s in (part.strip() for part in symbols.split(",")) if s]
        if not requested:
            raise ApiError(422, "Provide at least one symbol", "no_symbols")
        batch = svc.market.get_quotes(requested)
        ordered = [batch.quotes[s] for s in (normalize_ticker(r) for r in requested) if s in batch.quotes]
        seen, unique = set(), []
        for q in ordered:
            if q.symbol not in seen:
                seen.add(q.symbol)
                unique.append(q)
        return QuotesOut(quotes=unique, errors=batch.errors, notices=batch.notices,
                         data_status=combine_status([q.data_status for q in unique]))

    @router.get("/trend/{symbol}", response_model=TrendOut, responses={404: {"description": "Unknown symbol"}, **TOO_MANY},
                summary="Price history and trend summary for one symbol",
                description="Closing prices (oldest first) for charting, plus moving averages, 1- and 3-month change, volatility, and "
                            "largest drop. Descriptions of the past only; not a forecast.")
    def trend(request: Request, symbol: str = Path(min_length=1, max_length=12), days: int = Query(default=180, ge=30, le=365),
              svc: ApiServices = Depends(get_services)) -> TrendOut:
        enforce(svc.limiters["read"], client_key(request))
        sym = normalize_ticker(symbol)
        if sym is None:
            raise ApiError(422, "Not a valid ticker symbol", "invalid_symbol")
        summary, history = svc.market.get_trend(sym, days)
        return TrendOut(symbol=sym, trend=summary, points=history.points, source=history.source, data_status=history.status,
                        notice=history.notice)

    @router.get("/context/{symbol}", response_model=ContextOut, responses={404: {"description": "Unknown symbol"}, **TOO_MANY},
                summary="A symbol in context: compared with a benchmark, with a plain-language reading",
                description="Adds to /trend: the same period for a benchmark (default SPY, an S&P 500 proxy), the percentage-point gaps, both "
                            "series rebased to 100 for one chart, and `insights` written from the computed numbers only. Descriptive, not a forecast. "
                            "If the benchmark is unavailable the symbol's own reading is still returned, with a notice.")
    async def context(request: Request, symbol: str = Path(min_length=1, max_length=12), benchmark: str = Query(default=DEFAULT_BENCHMARK, max_length=12),
                      days: int = Query(default=180, ge=30, le=365), svc: ApiServices = Depends(get_services)) -> ContextOut:
        enforce(svc.limiters["read"], client_key(request))
        sym, bench_sym = normalize_ticker(symbol), normalize_ticker(benchmark)
        if sym is None or bench_sym is None:
            raise ApiError(422, "Not a valid ticker symbol", "invalid_symbol")
        notices: list[str] = []
        summary, history = await asyncio.to_thread(svc.market.get_trend, sym, days)
        statuses, series = [history.status], {sym: history.points}
        bench_summary = None
        if bench_sym != sym:
            try:
                bench_summary, bench_history = await asyncio.to_thread(svc.market.get_trend, bench_sym, days)
                statuses.append(bench_history.status)
                series[bench_sym] = bench_history.points
            except Exception:  # the benchmark is a nicety: never lose the symbol's own answer
                notices.append(f"The comparison with {bench_sym} is unavailable right now.")
        notices += [n for n in (history.notice, bench_summary.notice if bench_summary else None) if n]
        return ContextOut(
            symbol=sym, benchmark=bench_sym if bench_summary else None, benchmark_label=bench_sym if bench_summary else None, trend=summary,
            benchmark_trend=bench_summary, relative=relative_performance(summary, bench_summary) if bench_summary else {},
            insights=interpret(summary, bench_summary, bench_sym if bench_summary else None), comparison=rebased(series),
            data_status=combine_status(statuses), notices=list(dict.fromkeys(notices)))

    @router.get("/overview", response_model=OverviewOut, responses=TOO_MANY,
                summary="How the main market segments have moved (US large/small/tech, international, bonds)",
                description="Uses ETFs as proxies (SPY, QQQ, IWM, VXUS, BND). Each segment carries its own data status; `takeaways` compare the "
                            "segments in plain language using the computed returns only.")
    async def overview(request: Request, svc: ApiServices = Depends(get_services)) -> OverviewOut:
        enforce(svc.limiters["read"], client_key(request))

        async def one(symbol: str, label: str, description: str) -> SegmentOut:
            try:
                trend, history = await asyncio.to_thread(svc.market.get_trend, symbol, 120)
            except Exception:
                return SegmentOut(symbol=symbol, label=label, description=description, error="Data unavailable right now")
            return SegmentOut(symbol=symbol, label=label, description=description, last_close=trend.last_close, return_1m_pct=trend.return_1m_pct,
                              return_3m_pct=trend.return_3m_pct, volatility_annual_pct=trend.volatility_annual_pct, trend_label=trend.trend_label,
                              data_status=history.status, source=history.source)

        segments = list(await asyncio.gather(*(one(*s) for s in SEGMENTS)))
        ok = [s for s in segments if s.error is None]
        notices = [f"{s.label} ({s.symbol}): data unavailable right now." for s in segments if s.error]
        phrases = {DataStatus.cached: "a saved copy rather than a fresh lookup", DataStatus.stale: "the last known values, which may be out of date",
                   DataStatus.sample: "SAMPLE data, not real prices"}
        for status, phrase in phrases.items():  # one sentence per kind of data, not one per segment
            names = [s.symbol for s in ok if s.data_status is status]
            if names:
                notices.append(f"{', '.join(names[:-1])} and {names[-1]} are {phrase}." if len(names) > 1 else f"{names[0]} is {phrase}.")
        return OverviewOut(segments=segments, data_status=combine_status([s.data_status for s in ok]) if ok else DataStatus.live,
                           notices=notices, takeaways=_takeaways(ok))

    @router.get("/status", response_model=list[ProviderStatusOut], summary="Market-data provider health and remaining quota")
    def status(request: Request, svc: ApiServices = Depends(get_services)) -> list[ProviderStatusOut]:
        enforce(svc.limiters["read"], client_key(request))
        return [ProviderStatusOut(**{k: v for k, v in item.items() if k in ProviderStatusOut.model_fields}) for item in svc.market.status()]

    return router


def _takeaways(segments: list[SegmentOut]) -> list[str]:
    """Cross-segment reading from computed 3-month returns (falls back to 1-month). Descriptive only."""
    key = "return_3m_pct" if sum(s.return_3m_pct is not None for s in segments) >= 2 else "return_1m_pct"
    span = "3 months" if key == "return_3m_pct" else "1 month"
    ranked = sorted((s for s in segments if getattr(s, key) is not None), key=lambda s: getattr(s, key), reverse=True)
    if len(ranked) < 2:
        return []
    top, bottom = ranked[0], ranked[-1]
    out = [f"Over {span}, {top.label} led ({getattr(top, key):+.1f}%) and {bottom.label} lagged ({getattr(bottom, key):+.1f}%)."]
    stocks = [s for s in ranked if s.symbol != "BND"]
    bonds = next((s for s in ranked if s.symbol == "BND"), None)
    if bonds is not None and stocks:
        avg = sum(getattr(s, key) for s in stocks) / len(stocks)
        b = getattr(bonds, key)
        if (avg >= 0) != (b >= 0):
            out.append("Stocks and bonds moved in opposite directions, which is one reason people hold both.")
        else:
            out.append("Stocks and bonds moved in the same direction over this period, so mixing them did not offset each other this time.")
    out.append("Segments are shown through ETF proxies, and past movement does not predict what comes next.")
    return out
