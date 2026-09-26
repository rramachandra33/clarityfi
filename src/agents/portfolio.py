"""Portfolio Analysis agent: values a user's holdings, scores diversification and risk, explains the result.

Numbers come from `compute_portfolio_metrics` (pure code) and live-ish prices from MarketService; the
LLM only narrates. Policy for prices:
- real quotes (live / cached / last-known) are used, and any non-live status is disclosed;
- SAMPLE quotes are NEVER used to value a portfolio (they would show invented values as real):
  those holdings fall back to your cost basis, and the answer says so.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agents.base import AgentRequest, BaseAgent
from src.agents.narration import narrate
from src.core.disclaimers import with_data_notices
from src.core.freshness import DataStatus, combine_status, provider_label
from src.core.models import AgentName, AgentResult, Portfolio, PortfolioMetrics, RiskTolerance
from src.core.portfolio_calc import compute_portfolio_metrics
from src.core.tracing import observability_callbacks
from src.data.db import Database
from src.data.repositories import PortfolioRepository, UserRepository

NEEDS_PORTFOLIO = (
    "I don't see any holdings for you yet. Add the funds or stocks you own (ticker and number of shares), or a cash balance, "
    "and I can explain how diversified your portfolio is, what it costs in fees, and how risky the mix looks."
)

SCORE_EXPLANATIONS = {
    "diversification_score": "0-100: 45% how many independent positions you effectively hold, 35% how spread across asset classes "
                             "(stocks, bonds, cash...), 20% how large your biggest single-company position is. Broad index funds "
                             "count as many holdings.",
    "risk_score": "0-100: value-weighted risk of the asset classes held (cash lowest, crypto highest), plus a small add-on when one "
                  "position dominates. A label such as 'Growth' describes the mix, it is not a forecast.",
}


@dataclass
class PortfolioAnalysis:
    """Deterministic result of pricing + scoring a portfolio (no LLM involved)."""

    portfolio: Portfolio
    metrics: PortfolioMetrics | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    status: DataStatus = DataStatus.live
    priced_with: list[str] = field(default_factory=list)
    unpriced: list[str] = field(default_factory=list)
    sample_ignored: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    stated_risk: RiskTolerance | None = None
    empty: bool = False


class PortfolioAgent(BaseAgent):
    name = AgentName.portfolio
    description = "Analyzes a portfolio: value, allocation, diversification, fees, and risk, in plain language."

    def __init__(self, db: Database, market_service=None, *, llm_factory: Callable[[], Any] | None = None,
                 callbacks_factory: Callable[[], list[Any]] = observability_callbacks) -> None:
        self.db = db
        self._market = market_service
        self._llm_factory = llm_factory
        self._callbacks_factory = callbacks_factory

    @property
    def market(self):
        if self._market is None:
            from src.data.market_client import get_market_service

            self._market = get_market_service()
        return self._market

    def _load(self, request: AgentRequest) -> tuple[Portfolio, RiskTolerance | None]:
        profile = UserRepository(self.db).get(request.user_id)
        given = request.context.get("portfolio")
        portfolio = (given if isinstance(given, Portfolio) else Portfolio.model_validate({"user_id": request.user_id, **given})
                     if given else PortfolioRepository(self.db).get(request.user_id))
        return portfolio, (profile.risk_tolerance if profile else None)

    @staticmethod
    def _template(facts: dict[str, Any]) -> str:
        m = facts
        lines = [f"Your portfolio is worth about ${m['total_value']:,.2f} "
                 f"(${m['invested_value']:,.2f} invested and ${m['cash_value']:,.2f} in cash)."]
        if m["allocation_by_asset_class"]:
            mix = ", ".join(f"{k.replace('_', ' ')} {v:.1f}%" for k, v in m["allocation_by_asset_class"].items())
            lines.append(f"By asset class: {mix}.")
        if m["top_holdings"]:
            lines.append("Largest holdings: " + ", ".join(f"{h['ticker']} {h['weight_pct']:.1f}%" for h in m["top_holdings"]) + ".")
        lines.append(f"Diversification score: {m['diversification_score']:.0f} out of 100 (about {m['effective_positions']:.1f} "
                     f"effective positions). Risk mix: {m['risk_label']} (risk score {m['risk_score']:.0f} out of 100).")
        if m["unrealized_gain_pct"] is not None:
            lines.append(f"Compared with what you paid, your investments are {m['unrealized_gain_pct']:+.1f}%.")
        if m["weighted_expense_ratio_pct"] is not None:
            lines.append(f"Estimated fund fees: {m['weighted_expense_ratio_pct']:.3f}% a year (about ${m['annual_fee_estimate']:,.2f}).")
        lines.extend(m["observations"])
        return "\n".join(lines)

    async def analyze(self, user_id: str, portfolio: Portfolio | dict | None = None) -> "PortfolioAnalysis":
        """Price and score a portfolio with NO LLM: shared by the agent (which then narrates) and the REST API.

        `portfolio` overrides the saved one (e.g. what-if analysis). SAMPLE quotes are never used for valuation."""
        request = AgentRequest(question="", user_id=user_id, context={"portfolio": portfolio} if portfolio else {})
        portfolio_obj, stated_risk = await asyncio.to_thread(self._load, request)
        if not portfolio_obj.holdings and portfolio_obj.cash <= 0:
            return PortfolioAnalysis(portfolio=portfolio_obj, empty=True, stated_risk=stated_risk)

        tickers = list(dict.fromkeys(h.ticker for h in portfolio_obj.holdings))
        batch = await asyncio.to_thread(self.market.get_quotes, tickers) if tickers else None
        quotes = batch.quotes if batch else {}
        real = {t: q for t, q in quotes.items() if q.data_status is not DataStatus.sample}
        sample_only = sorted(t for t, q in quotes.items() if q.data_status is DataStatus.sample)
        unpriced = sorted(set(tickers) - set(real))

        notices = list(dict.fromkeys(q.notice for q in real.values() if q.notice))
        if unpriced:
            notices.append(f"Live prices for {', '.join(unpriced)} were unavailable, so those holdings are valued at your cost basis "
                           "(or $0 if no cost basis is known).")
        statuses = [q.data_status for q in real.values()] + ([DataStatus.stale] if unpriced else [])
        metrics = compute_portfolio_metrics(portfolio_obj, {t: q.price for t, q in real.items()}, stated_risk=stated_risk)
        return PortfolioAnalysis(
            portfolio=portfolio_obj, metrics=metrics, rows=self._rows(portfolio_obj, real, metrics), notices=notices,
            status=combine_status(statuses), priced_with=sorted(real), unpriced=unpriced, sample_ignored=sample_only,
            errors=batch.errors if batch else {}, stated_risk=stated_risk)

    async def _execute(self, request: AgentRequest) -> AgentResult:
        a = await self.analyze(request.user_id, request.context.get("portfolio"))
        if a.empty:
            return AgentResult(agent=self.name, content=NEEDS_PORTFOLIO, data={"needs_portfolio": True})
        metrics, rows, notices = a.metrics, a.rows, a.notices
        facts = {
            "total_value": metrics.total_value, "invested_value": metrics.invested_value, "cash_value": metrics.cash_value,
            "allocation_by_asset_class": metrics.allocation_by_asset_class,
            "top_holdings": [{"ticker": r["ticker"], "weight_pct": r["weight_pct"]} for r in rows[:5]],
            "diversification_score": metrics.diversification_score, "effective_positions": metrics.effective_positions,
            "risk_score": metrics.risk_score, "risk_label": metrics.risk_label,
            "stated_risk_tolerance": a.stated_risk.value if a.stated_risk else None,
            "unrealized_gain_pct": metrics.unrealized_gain_pct, "weighted_expense_ratio_pct": metrics.weighted_expense_ratio,
            "annual_fee_estimate": metrics.annual_fee_estimate, "observations": metrics.notes, "how_scores_work": SCORE_EXPLANATIONS,
            "data_notices": notices,
        }
        text, mode = await narrate(request.question, facts, lambda: self._template(facts),
                                   llm_factory=self._llm_factory, callbacks_factory=self._callbacks_factory, max_words=260,
                                   agent_name=self.name.value)
        return AgentResult(
            agent=self.name, content=with_data_notices(text, notices),
            data_status=a.status, notice=" ".join(notices) or None,
            data={"metrics": metrics.model_dump(mode="json"), "holdings": rows, "priced_with": a.priced_with,
                  "unpriced": a.unpriced, "sample_quotes_ignored": a.sample_ignored, "notices": notices, "narration": mode,
                  "errors": a.errors})

    @staticmethod
    def _rows(portfolio: Portfolio, real: dict, metrics: PortfolioMetrics) -> list[dict[str, Any]]:
        """Per-holding table for the dashboard, largest first."""
        rows = []
        for h in portfolio.holdings:
            q = real.get(h.ticker)
            value = h.shares * q.price if q else (h.cost_basis or 0.0)
            rows.append({"ticker": h.ticker, "shares": h.shares, "asset_class": h.asset_class.value,
                         "price": round(q.price, 2) if q else None, "value": round(value, 2),
                         "weight_pct": metrics.allocation_by_holding.get(h.ticker, 0.0),
                         "price_source": provider_label(q.source) if q else "cost basis",
                         "price_status": q.data_status.value if q else "unavailable"})
        return sorted(rows, key=lambda r: -r["value"])
