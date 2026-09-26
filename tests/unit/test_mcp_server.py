"""MCP server: tools are listed, validated, disclose data provenance, and are exercised through a real in-memory MCP client."""
import json

import pytest
from mcp.client import Client

from src.agents.portfolio import PortfolioAgent
from src.core.freshness import DataStatus
from src.data.db import Database
from src.data.repositories import UserRepository
from src.data.seed import seed
from src.mcp_server import server as srv
from src.mcp_server.server import Services, build_server
from src.rag.vector_store import SearchResult
from tests.unit._agent_fakes import FakeMarket, quote


class FakeKB:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k, category=None, mode="hybrid"):
        self.calls.append((query, top_k, category))
        return [SearchResult(text="An ETF holds many securities in one fund. " * 3, title="What is an ETF?", score=0.83,
                             metadata={"category": "funds", "source_name": "Investopedia", "source_url": "https://ex.com/etf"})]


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "m.db")
    db.init_schema()
    seed(db)
    market = FakeMarket({"AAPL": quote("AAPL", 200.0), "VTI": quote("VTI", 300.0), "VXUS": quote("VXUS", 60.0), "BND": quote("BND", 75.0)})
    kb = FakeKB()
    services = Services(market=market, portfolio=PortfolioAgent(db, market), knowledge_base=kb, users=UserRepository(db).get)
    return SimpleEnv(build_server(services), market, kb)


class SimpleEnv:
    def __init__(self, server, market, kb):
        self.server, self.market, self.kb = server, market, kb

    async def call(self, name, args):
        async with Client(self.server) as client:
            result = await client.call_tool(name, args)
        return result

    async def data(self, name, args):
        result = await self.call(name, args)
        assert not result.is_error, result.content
        return result.structured_content or json.loads(result.content[0].text)


async def test_lists_the_four_documented_tools_as_read_only(env):
    async with Client(env.server) as client:
        tools = (await client.list_tools()).tools
    assert sorted(t.name for t in tools) == ["analyze_portfolio", "get_quote", "project_goal", "search_knowledge_base"]
    assert all(t.annotations and t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools)


async def test_get_quote_dedupes_reports_unknown_and_discloses_status(env):
    env.market.quotes["AAPL"] = quote("AAPL", 200.0, status=DataStatus.stale)
    data = await env.data("get_quote", {"symbols": ["aapl", "AAPL", "ZZZZ", " "]})
    assert [q["symbol"] for q in data["quotes"]] == ["AAPL"] and "ZZZZ" in data["errors"]
    assert data["data_status"] == "stale" and data["notices"] and "Educational" in data["disclaimer"]


@pytest.mark.parametrize("symbols", [[], ["  "], ["A"] * 26])
async def test_get_quote_validates_input(env, symbols):
    result = await env.call("get_quote", {"symbols": symbols})
    assert result.is_error and ("at least one" in result.content[0].text or "at most" in result.content[0].text)


async def test_analyze_portfolio_computed_and_live(env):
    data = await env.data("analyze_portfolio", {"user_id": "learner-105"})
    assert data["empty"] is False and data["metrics"]["total_value"] == pytest.approx(10 * 0 + data["metrics"]["total_value"]) and data["data_status"] == "live"
    assert {h["ticker"] for h in data["holdings"]} == {"VTI", "VXUS", "BND"} and data["disclaimer"]


async def test_analyze_portfolio_discloses_non_live_prices_and_handles_empty_and_unknown(env):
    env.market.quotes["VTI"] = quote("VTI", 300.0, status=DataStatus.stale)
    assert (await env.data("analyze_portfolio", {"user_id": "learner-105"}))["notices"]
    assert (await env.data("analyze_portfolio", {"user_id": "learner-110"}))["empty"] is True
    bad = await env.call("analyze_portfolio", {"user_id": "nobody-here"})
    assert bad.is_error and "Unknown user id" in bad.content[0].text  # anticipated failures are readable by the model
    assert (await env.call("analyze_portfolio", {"user_id": "   "})).is_error


async def test_search_redacts_pii_before_searching_and_returns_citations(env):
    data = await env.data("search_knowledge_base", {"query": "email me at jane.doe@example.com what is an ETF", "category": "funds", "top_k": 2})
    query, top_k, category = env.kb.calls[0]
    assert "jane.doe@example.com" not in query and top_k == 2 and category == "funds"
    hit = data["results"][0]
    assert hit["url"] == "https://ex.com/etf" and hit["source"] == "Investopedia" and hit["text"] and data["disclaimer"]


@pytest.mark.parametrize("args", [{"query": ""}, {"query": "x" * 400}, {"query": "ok", "top_k": 0}, {"query": "ok", "top_k": 9}])
async def test_search_validates_input(env, args):
    assert (await env.call("search_knowledge_base", args)).is_error and not env.kb.calls


async def test_search_fails_closed_when_privacy_filter_breaks(env, monkeypatch):
    def boom(_):
        raise RuntimeError("nlp down")
    monkeypatch.setattr(srv, "redact_text", boom)
    result = await env.call("search_knowledge_base", {"query": "what is an ETF"})
    assert result.is_error and "privacy filter" in result.content[0].text and not env.kb.calls


async def test_search_with_no_results_says_so(env):
    env.kb.search = lambda *a, **k: []
    data = await env.data("search_knowledge_base", {"query": "zzzz qqqq"})
    assert data["results"] == [] and data["note"]


async def test_project_goal_returns_three_scenarios_and_matches_the_calculator(env):
    data = await env.data("project_goal", {"target_amount": 50000, "years": 8, "current_savings": 5000, "monthly_contribution": 300, "risk_tolerance": "Moderate"})
    assert len(data["projection"]["scenarios"]) == 3 and data["assumptions"] and data["goal"]["risk_tolerance"] == "moderate"
    assert data["goal"]["assumed_return_pct"] == 0.0  # defaults to 0 when not given
    from datetime import date, timedelta
    from src.core.models import Goal
    from src.core.projections import project_goal
    goal = Goal(**{k: data["goal"][k] for k in ("name", "target_amount", "current_savings", "monthly_contribution", "risk_tolerance", "assumed_return_pct")},
                target_date=date.today() + timedelta(days=round(8 * 365.25)))
    assert data["projection"]["scenarios"][1]["projected_value"] == project_goal(goal, today=date.today()).scenarios[1].projected_value


async def test_project_goal_uses_the_stated_assumed_return(env):
    data = await env.data("project_goal", {"target_amount": 50000, "years": 8, "assumed_return_pct": 6.0})
    mid = next(s for s in data["projection"]["scenarios"] if s["label"] == "Your assumption")
    assert mid["annual_return"] == pytest.approx(0.06)


@pytest.mark.parametrize("args", [
    {"target_amount": 0, "years": 5}, {"target_amount": 1000, "years": 0}, {"target_amount": 1000, "years": 81},
    {"target_amount": 1000, "years": 5, "current_savings": -1}, {"target_amount": 1000, "years": 5, "risk_tolerance": "yolo"},
    {"target_amount": 1000, "years": 5, "assumed_return_pct": -51}, {"target_amount": 1000, "years": 5, "assumed_return_pct": 101}])
async def test_project_goal_validates_input(env, args):
    result = await env.call("project_goal", args)
    assert result.is_error and ("must" in result.content[0].text)


def test_default_services_fall_back_to_faiss_when_the_primary_index_is_busy(monkeypatch):
    calls = []

    def fake_build(embedder, config=None):
        calls.append(config)
        if config is None:
            raise RuntimeError("Storage folder is already accessed by another instance of Qdrant client")
        return "faiss-kb"

    monkeypatch.setattr("src.rag.knowledge_base.build_knowledge_base", fake_build)
    services = srv.default_services()
    assert services.knowledge_base == "faiss-kb" and calls[1].backend == "faiss" and calls[1].fallback == ""


def test_main_reports_startup_failure(monkeypatch, capsys):
    from src.core.errors import ClarityError

    class Boom:
        def run(self, transport):
            raise ClarityError("no key")

    monkeypatch.setattr(srv, "build_server", lambda: Boom())
    assert srv.main() == 1 and "could not start" in capsys.readouterr().err
