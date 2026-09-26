"""Market Analysis agent: live quotes and (on request) price-history trends, with honest data labelling.

Code fetches and computes everything (MarketService: Alpha Vantage -> yFinance -> last-good cache ->
sample data); the LLM only explains. Every cached / last-known / SAMPLE figure is disclosed in
`AgentResult.notice` and in the answer text. Descriptions of the past, never predictions.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from src.agents.base import AgentRequest, BaseAgent
from src.agents.entities import extract_symbols, proxy_notes, wants_trend
from src.agents.narration import narrate
from src.core.disclaimers import MARKET_DATA_DISCLAIMER, with_data_notices
from src.core.freshness import DataStatus, combine_status, provider_label
from src.core.models import AgentName, AgentResult, Quote, TrendSummary
from src.core.tracing import observability_callbacks
from src.utils.validators import normalize_ticker

MAX_TREND_SYMBOLS = 3
MARKET_RULES = (
    "- Each price is as of its price_is_as_of_trading_day (the market session it belongs to), which may be earlier than today; "
    "never say 'today' or 'now' unless that is clearly the case. previous_session_close belongs to the session BEFORE that day.\n"
    "- Trend figures describe the past only."
)
ASK_FOR_SYMBOLS = (
    "Which stock, fund, or crypto would you like to look up? Give me a ticker (like AAPL or VTI) or a well-known name "
    "(like Apple, bitcoin, or the S&P 500)."
)


def _quote_facts(q: Quote) -> dict[str, Any]:
    return {"symbol": q.symbol, "price": round(q.price, 2), "change_pct": None if q.change_pct is None else round(q.change_pct, 2),
            "previous_session_close": None if q.previous_close is None else round(q.previous_close, 2),
            "price_is_as_of_trading_day": q.trading_day.isoformat() if q.trading_day else "latest available", "source": provider_label(q.source),
            "data_status": q.data_status.value}


def _trend_facts(t: TrendSummary) -> dict[str, Any]:
    return {"symbol": t.symbol, "label": t.trend_label, "last_close": t.last_close, "sma_20": t.sma_20, "sma_50": t.sma_50,
            "return_1m_pct": t.return_1m_pct, "return_3m_pct": t.return_3m_pct, "volatility_annual_pct": t.volatility_annual_pct,
            "max_drawdown_pct": t.max_drawdown_pct, "period_high": t.period_high, "period_low": t.period_low,
            "trading_days_analyzed": t.points, "data_status": t.data_status.value}


def _fmt_change(pct: float | None) -> str:
    return "" if pct is None else f" ({pct:+.2f}% vs the previous close)"


class MarketAgent(BaseAgent):
    name = AgentName.market
    description = "Looks up quotes and past-price trends for stocks, funds, and crypto, and explains them."

    def __init__(self, market_service=None, *, llm_factory: Callable[[], Any] | None = None,
                 callbacks_factory: Callable[[], list[Any]] = observability_callbacks) -> None:
        self._market = market_service
        self._llm_factory = llm_factory
        self._callbacks_factory = callbacks_factory

    @property
    def market(self):
        if self._market is None:
            from src.data.market_client import get_market_service

            self._market = get_market_service()
        return self._market

    # -- deterministic text (also the fallback when the LLM is unavailable or ungrounded) -----------
    @staticmethod
    def _template(facts: dict[str, Any]) -> str:
        lines = []
        for q in facts["quotes"]:
            day = q["price_is_as_of_trading_day"]
            when = f", as of {day}" if day != "latest available" else ""
            lines.append(f"{q['symbol']}: ${q['price']:,.2f}{_fmt_change(q['change_pct'])} (source: {q['source']}{when}).")
        for t in facts["trends"]:
            parts = [f"{t['symbol']} trend over the last {t['trading_days_analyzed']} trading days: {t['label']}."]
            if t["return_1m_pct"] is not None:
                parts.append(f"1-month change {t['return_1m_pct']:+.2f}%.")
            if t["return_3m_pct"] is not None:
                parts.append(f"3-month change {t['return_3m_pct']:+.2f}%.")
            if t["volatility_annual_pct"] is not None:
                parts.append(f"Annualized volatility about {t['volatility_annual_pct']:.1f}%.")
            if t["max_drawdown_pct"] is not None:
                parts.append(f"Largest drop from a peak: {t['max_drawdown_pct']:.1f}%.")
            lines.append(" ".join(parts))
        if facts["unknown_symbols"]:
            lines.append("I couldn't find: " + ", ".join(facts["unknown_symbols"]) + ".")
        lines.extend(facts["index_proxies"])
        lines.append("Trends describe past prices only and do not predict future results. " + MARKET_DATA_DISCLAIMER)
        return "\n".join(lines)

    async def _execute(self, request: AgentRequest) -> AgentResult:
        raw = request.context.get("symbols")
        symbols = [s for s in (normalize_ticker(x) for x in raw) if s] if raw else extract_symbols(request.question)
        if not symbols:
            return AgentResult(agent=self.name, content=ASK_FOR_SYMBOLS, data={"needs_symbols": True})

        batch = await asyncio.to_thread(self.market.get_quotes, symbols)
        quotes = [batch.quotes[s] for s in symbols if s in batch.quotes]
        unknown = [s for s in symbols if s not in batch.quotes]

        trends: list[TrendSummary] = []
        if quotes and (request.context.get("include_trend") or wants_trend(request.question)):
            for q in quotes[:MAX_TREND_SYMBOLS]:
                try:
                    trend, _history = await asyncio.to_thread(self.market.get_trend, q.symbol)
                    trends.append(trend)
                except Exception:  # a failed trend must not lose the quote we already have
                    continue

        notices = list(dict.fromkeys([*batch.notices, *[t.notice for t in trends if t.notice]]))
        status = combine_status([q.data_status for q in quotes] + [t.data_status for t in trends])
        facts = {
            "quotes": [_quote_facts(q) for q in quotes], "trends": [_trend_facts(t) for t in trends],
            "unknown_symbols": unknown, "index_proxies": proxy_notes(symbols, request.question), "data_notices": notices,
        }
        if not quotes:
            content = with_data_notices(self._template(facts), notices)
            return AgentResult(agent=self.name, content=content, data={"errors": batch.errors, "symbols": symbols, "narration": "template"})

        text, mode = await narrate(request.question, facts, lambda: self._template(facts), extra_rules=MARKET_RULES,
                                   llm_factory=self._llm_factory, callbacks_factory=self._callbacks_factory, agent_name=self.name.value)
        return AgentResult(
            agent=self.name, content=with_data_notices(text, notices), data_status=status,
            notice=" ".join(notices) or None,
            data={"symbols": symbols, "quotes": [q.model_dump(mode="json") for q in quotes],
                  "trends": [t.model_dump(mode="json") for t in trends], "unknown_symbols": unknown,
                  "errors": batch.errors, "notices": notices, "narration": mode})
