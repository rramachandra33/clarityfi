"""ClarityFI MCP server: four read-only educational tools for any MCP client (Claude Desktop, IDEs, agents).

Tools: get_quote · analyze_portfolio · search_knowledge_base · project_goal

    python -m src.mcp_server.server            # stdio transport (what Claude Desktop launches)

The tools reuse the same services and safeguards as the web API:
- numbers come from tested code (market chain, portfolio and goal calculators), never from a model;
- every result says where the data came from (`data_status`: live / cached / stale / sample) and carries a plain
  `notice` for anything that is not live;
- free text (search queries) is validated and PII-redacted before it can reach an embedding provider (fail closed);
- every result carries the education-only disclaimer. Nothing here writes data.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from src.agents.goal_planning import GoalPlanningAgent
from src.core.disclaimers import PLAIN_DISCLAIMER
from src.core.errors import ClarityError, GuardrailError
from src.core.freshness import combine_status
from src.core.models import Goal, RiskTolerance
from src.core.pii import redact_text
from src.core.projections import project_goal
from src.rag.knowledge_base import to_citations
from src.utils.validators import normalize_ticker, validate_user_input

logger = logging.getLogger("clarityfi.mcp")

MAX_SYMBOLS = 25
MAX_QUERY_CHARS = 300
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)
LOCAL_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

INSTRUCTIONS = (
    "ClarityFI provides financial EDUCATION tools. Results are illustrations and general information, not personalized "
    "financial, investment, legal or tax advice. Always relay `notice` text to the user when it is present: it means the "
    "data is a saved copy, out of date, or SAMPLE data. Never present sample data as real."
)


@dataclass
class Services:
    """What the tools need; tests pass fakes, `default_services()` builds the real ones."""
    market: Any        # MarketService: get_quotes(symbols)
    portfolio: Any     # PortfolioAgent: async analyze(user_id)
    knowledge_base: Any  # KnowledgeBase: search(query, top_k, category=)
    users: Any         # callable(user_id) -> UserProfile | None


def default_services() -> Services:
    from src.agents.portfolio import PortfolioAgent
    from src.core.config import settings
    from src.core.openai_client import OpenAIClient
    from src.data.db import Database
    from src.data.market_client import get_market_service
    from src.data.repositories import UserRepository
    from src.rag.embedding import CachingEmbedder
    from src.rag.knowledge_base import build_knowledge_base

    db = Database(settings.app_db_path)
    db.init_schema()
    market = get_market_service()
    embedder = CachingEmbedder(OpenAIClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url,
                                            chat_model=settings.openai_chat_model, embed_model=settings.openai_embed_model))
    try:
        kb = build_knowledge_base(embedder)
    except Exception:  # embedded Qdrant allows one process; if the web API holds it, search with the FAISS index instead
        from src.core.config import RagConfig

        logger.warning("Primary index is busy; the MCP server will search the FAISS index")
        kb = build_knowledge_base(embedder, RagConfig(backend="faiss", fallback=""))
    return Services(market=market, portfolio=PortfolioAgent(db, market), knowledge_base=kb, users=UserRepository(db).get)


def _with_disclaimer(payload: dict[str, Any]) -> dict[str, Any]:
    payload["disclaimer"] = PLAIN_DISCLAIMER
    return payload


def build_server(services: Services | None = None) -> MCPServer:
    """Create the server. `services=None` builds the real services lazily on the first tool call."""
    holder: dict[str, Services] = {}

    def svc() -> Services:
        if services is not None:
            return services
        if "real" not in holder:
            holder["real"] = default_services()
        return holder["real"]

    server = MCPServer("clarityfi", instructions=INSTRUCTIONS)

    @server.tool(annotations=READ_ONLY)
    async def get_quote(symbols: list[str]) -> dict[str, Any]:
        """Latest price and daily change for up to 25 tickers (stocks, ETFs, crypto like BTC-USD).

        Source order: Alpha Vantage, Yahoo Finance, last saved copy, then SAMPLE data (only if every provider is down).
        Each quote has `data_status` and, unless live, a `notice` to show the user. Unknown symbols are listed in `errors`.
        """
        cleaned = [s for s in (str(x).strip() for x in symbols or []) if s]
        if not cleaned:
            raise ToolError("Provide at least one ticker symbol, for example ['AAPL', 'VTI'].")
        if len(cleaned) > MAX_SYMBOLS:
            raise ToolError(f"Ask for at most {MAX_SYMBOLS} symbols at a time.")
        batch = await asyncio.to_thread(svc().market.get_quotes, cleaned)
        seen: set[str] = set()
        quotes = []
        for raw in cleaned:
            sym = normalize_ticker(raw)
            if sym and sym in batch.quotes and sym not in seen:
                seen.add(sym)
                quotes.append(batch.quotes[sym].model_dump(mode="json"))
        return _with_disclaimer({
            "quotes": quotes, "errors": dict(batch.errors), "notices": list(batch.notices),
            "data_status": combine_status([batch.quotes[s].data_status for s in seen]).value if seen else "live"})

    @server.tool(annotations=LOCAL_ONLY)
    async def analyze_portfolio(user_id: str) -> dict[str, Any]:
        """Analyze a ClarityFI user's saved portfolio: total value, allocation, diversification score, fees, risk label and notes.

        Use one of the demo ids learner-101 … learner-110 (or an id registered in the app). Values are computed in code;
        any price that is not live is disclosed in `notices`, and SAMPLE prices are never used for valuation.
        """
        try:
            uid = validate_user_input(user_id, 120)
        except GuardrailError as exc:
            raise ToolError(str(exc)) from exc
        profile = svc().users(uid)
        if profile is None:
            raise ToolError("Unknown user id. Try a demo id such as learner-105.")
        a = await svc().portfolio.analyze(profile.user_id)
        if a.empty:
            return _with_disclaimer({"user_id": profile.user_id, "empty": True, "message": "This user has no holdings or cash yet."})
        return _with_disclaimer({
            "user_id": profile.user_id, "empty": False, "metrics": a.metrics.model_dump(mode="json"), "holdings": a.rows,
            "data_status": a.status.value, "notices": a.notices, "priced_with": a.priced_with, "unpriced": a.unpriced,
            "stated_risk_tolerance": a.stated_risk.value if a.stated_risk else None})

    @server.tool(annotations=READ_ONLY)
    async def search_knowledge_base(query: str, category: str | None = None, top_k: int = 3) -> dict[str, Any]:
        """Search ClarityFI's curated 230-article finance education knowledge base (hybrid keyword + meaning search).

        Returns passages with title, publisher, link (when known), category and a relevance score. `category` may be a
        group such as 'tax', 'retirement', 'bonds', 'funds', 'portfolio', 'risk', 'basics', 'markets' or 'protection'.
        The query is checked and stripped of personal data before it is sent to the embedding provider.
        """
        try:
            cleaned = validate_user_input(query, MAX_QUERY_CHARS)
        except GuardrailError as exc:
            raise ToolError(str(exc)) from exc
        if not 1 <= top_k <= 8:
            raise ToolError("top_k must be between 1 and 8.")
        try:
            safe = (await asyncio.to_thread(redact_text, cleaned)).text
        except Exception as exc:  # fail closed: never send unfiltered text onward
            raise ToolError("The privacy filter is unavailable, so the search was not run.") from exc
        results = await asyncio.to_thread(lambda: svc().knowledge_base.search(safe, top_k, category=category or None))
        return _with_disclaimer({
            "query": safe, "category": category,
            "results": [{**c.model_dump(mode="json"), "text": r.text} for c, r in zip(to_citations(results), results)],
            "note": None if results else "No relevant passages found. Try different words."})

    @server.tool(name="project_goal", annotations=LOCAL_ONLY)
    async def project_goal_tool(target_amount: float, years: float, current_savings: float = 0.0, monthly_contribution: float = 0.0,
                                risk_tolerance: str = "moderate", goal_name: str = "My goal", assumed_return_pct: float = 0.0) -> dict[str, Any]:
        """Project a savings goal under three illustrative growth scenarios: below/expected/above, centered on your own
        `assumed_return_pct` (0% by default, meaning no assumed growth).

        Returns projected values (also in today's dollars), whether each scenario reaches the target, the TOTAL monthly amount
        needed, and a yearly path. Growth rates are assumptions for illustration, not forecasts. `risk_tolerance` is stored on
        the goal (it matters elsewhere, e.g. portfolio analysis) but does not affect these scenarios.
        """
        if not (0 < target_amount <= 1e10) or not (0 < years <= 80) or not (0 <= current_savings <= 1e10) or not (0 <= monthly_contribution <= 1e8):
            raise ToolError("target_amount must be positive; years between 0 and 80; savings and contributions must not be negative.")
        if not (-50 <= assumed_return_pct <= 100):
            raise ToolError("assumed_return_pct must be between -50 and 100.")
        try:
            risk = RiskTolerance(risk_tolerance.strip().lower())
        except ValueError as exc:
            raise ToolError("risk_tolerance must be conservative, moderate or aggressive.") from exc
        name = "".join(ch for ch in goal_name if ch.isprintable()).strip()[:80] or "My goal"
        goal = Goal(name=name, target_amount=target_amount, target_date=date.today() + timedelta(days=round(years * 365.25)),
                    current_savings=current_savings, monthly_contribution=monthly_contribution, risk_tolerance=risk,
                    assumed_return_pct=assumed_return_pct)
        projection = project_goal(goal, today=date.today())
        return _with_disclaimer({
            "goal": goal.model_dump(mode="json"), "projection": projection.model_dump(mode="json"),
            "assumptions": GoalPlanningAgent._facts(goal, projection, [])["assumptions"],
            "on_track_expected": projection.scenarios[1].on_track})

    return server


def main() -> int:
    # stdout carries the protocol, so logs must go to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        build_server().run(transport="stdio")
    except ClarityError as exc:
        print(f"ClarityFI MCP server could not start: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
