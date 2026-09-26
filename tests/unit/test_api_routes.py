"""REST layer: portfolio, market, goals, news, conversations, profile, error shape, rate limiting."""
import pytest
from fastapi.testclient import TestClient

from src.agents.goal_planning import GoalPlanningAgent
from src.agents.news import NewsAgent
from src.agents.portfolio import PortfolioAgent
from src.core.freshness import DataStatus
from src.core.models import NewsItem
from src.data.db import Database
from src.data.repositories import MessageRepository, SessionRepository
from src.data.seed import seed
from src.web_app.api import main
from src.web_app.api.services import ApiServices, default_limiters
from src.web_app.ratelimit import SlidingWindowLimiter
from src.workflow.memory import SqliteConversationMemory
from tests.unit._agent_fakes import FakeLLM, FakeMarket, quote

PRICES = {"VTI": 300.0, "VXUS": 60.0, "BND": 75.0, "AAPL": 200.0}


class FakeNews:
    def __init__(self, status="live", items=None):
        self.status, self.calls = status, []
        self.items = items if items is not None else [NewsItem(title="Rates hold steady", url="https://ex.com/1", source="Wire")]

    def search(self, query, limit=6):
        self.calls.append(query)
        notice = None if self.status == "live" else "Showing SAMPLE headlines."
        return self.items, {"source": "serpapi", "data_status": self.status, "notice": notice, "stale": False, "cache_hit": False}


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "api.db")
    db.init_schema()
    seed(db)
    market = FakeMarket({t: quote(t, p) for t, p in PRICES.items()})
    llm = FakeLLM("Explained.")
    news = FakeNews()
    svc = ApiServices(
        db=db, market=market, news=news,
        portfolio_agent=PortfolioAgent(db, market, llm_factory=lambda: llm, callbacks_factory=lambda: []),
        goal_agent=GoalPlanningAgent(db, llm_factory=lambda: llm, callbacks_factory=lambda: []),
        news_agent=NewsAgent(news, llm_factory=lambda: llm, callbacks_factory=lambda: []),
        memory=SqliteConversationMemory(db), limiters=default_limiters())
    main.app.dependency_overrides[main.get_services] = lambda: svc
    main.app.dependency_overrides[main.get_db] = lambda: db
    with TestClient(main.app) as client:
        client.svc, client.market, client.news = svc, market, news
        yield client
    main.app.dependency_overrides.clear()


HOLDINGS = {"holdings": [{"ticker": "vti", "shares": 10, "cost_basis": 250, "asset_class": "stock"}], "cash": 100}


def test_error_bodies_are_flat_with_a_code(env):
    r = env.get("/api/portfolio/nobody-here")
    assert r.status_code == 404 and r.json() == {"detail": "Unknown user id", "code": "user_not_found"}


def test_portfolio_get_put_roundtrip_and_new_user(env):
    assert env.get("/api/portfolio/learner-105").json()["synthetic"] is True
    r = env.put("/api/portfolio/brand-new-user", json=HOLDINGS)
    assert r.status_code == 200 and r.json()["holdings"][0]["ticker"] == "VTI" and r.json()["synthetic"] is False
    assert env.get("/api/portfolio/brand-new-user").json()["cash"] == 100


@pytest.mark.parametrize("body", [
    {"holdings": [{"ticker": "VTI", "shares": 1}, {"ticker": "vti", "shares": 2}], "cash": 0},
    {"holdings": [{"ticker": "VTI", "shares": -1}], "cash": 0},
    {"holdings": [{"ticker": "VTI", "shares": 1}] * 101, "cash": 0},
])
def test_portfolio_put_validation(env, body):
    assert env.put("/api/portfolio/learner-105", json=body).status_code == 422


def test_analysis_is_deterministic_and_discloses_non_live_prices(env):
    ok = env.get("/api/portfolio/learner-105/analysis").json()
    assert ok["metrics"]["total_value"] > 0 and ok["data_status"] == "live" and ok["explanation"] is None
    env.market.quotes["VTI"] = quote("VTI", 300.0, status=DataStatus.stale)
    stale = env.get("/api/portfolio/learner-105/analysis").json()
    assert stale["data_status"] == "stale" and stale["notice"] and stale["notices"]


def test_analysis_explain_uses_llm_and_empty_portfolio_is_flagged(env):
    assert env.get("/api/portfolio/learner-105/analysis?explain=true").json()["explanation"]
    env.put("/api/portfolio/empty-one", json={"holdings": [], "cash": 0})
    assert env.get("/api/portfolio/empty-one/analysis").json()["empty"] is True


def test_what_if_analysis_does_not_save(env):
    before = env.get("/api/portfolio/learner-105").json()
    r = env.post("/api/portfolio/learner-105/analysis", json=HOLDINGS)
    assert r.json()["what_if"] is True and r.json()["metrics"]["total_value"] == pytest.approx(3100.0)
    assert env.get("/api/portfolio/learner-105").json() == before


def test_market_quotes_partial_results_and_dedupe(env):
    r = env.get("/api/market/quotes?symbols=aapl,AAPL,ZZZZ,,VTI").json()
    assert [q["symbol"] for q in r["quotes"]] == ["AAPL", "VTI"] and "ZZZZ" in r["errors"]
    assert env.get("/api/market/quotes?symbols=%20,%20").status_code == 422


def test_market_quotes_carry_non_live_status(env):
    env.market.quotes["AAPL"] = quote("AAPL", 200.0, status=DataStatus.sample, source="sample")
    r = env.get("/api/market/quotes?symbols=AAPL").json()
    assert r["data_status"] == "sample" and r["quotes"][0]["notice"]


def test_trend_status_and_validation(env):
    ok = env.get("/api/market/trend/aapl?days=90").json()
    assert ok["symbol"] == "AAPL" and ok["points"] and ok["data_status"] == "live"
    assert env.get("/api/market/trend/AAPL?days=5").status_code == 422
    assert env.get("/api/market/trend/$$$").status_code == 422
    env.market.trend_status = DataStatus.stale
    assert env.get("/api/market/trend/AAPL").json()["notice"]


def test_provider_status(env):
    assert env.get("/api/market/status").json()[0]["provider"] == "alphavantage"


def test_goal_projection_variants(env):
    body = {"name": "House", "target_amount": 50000, "current_savings": 5000, "monthly_contribution": 300, "years": 8}
    r = env.post("/api/goals/project", json=body).json()
    # 3 scenarios centered on assumed_return_pct (0 by default)
    assert len(r["projection"]["scenarios"]) == 3 and r["explanation"] is None and r["assumptions"]
    assert r["projection"]["scenarios"][1]["label"] == "Your assumption" and r["projection"]["scenarios"][1]["annual_return"] == 0.0
    assert env.post("/api/goals/project", json={**body, "explain": True}).json()["explanation"]
    assert env.post("/api/goals/project", json={**body, "target_date": "2040-01-01"}).status_code == 422  # both given
    assert env.post("/api/goals/project", json={k: v for k, v in body.items() if k != "years"}).status_code == 422  # neither


def test_goal_save_list_delete_and_limits(env):
    body = {"name": "Car", "target_amount": 10000, "years": 3}
    saved = env.post("/api/goals/learner-105", json=body)
    assert saved.status_code == 201
    gid = saved.json()["id"]
    assert any(g["id"] == gid for g in env.get("/api/goals/learner-105").json())
    assert env.delete(f"/api/goals/learner-105/{gid}").status_code == 204
    gone = env.delete(f"/api/goals/learner-105/{gid}")
    assert gone.status_code == 404 and gone.json()["code"] == "goal_not_found"
    assert env.post("/api/goals/nobody-here", json=body).status_code == 404
    for i in range(25):  # distinct names, since same-name saves now update in place rather than adding a row
        last = env.post("/api/goals/learner-101", json={**body, "name": f"Car {i}"})
        if last.status_code == 409:
            break
    assert last.status_code == 409 and last.json()["code"] == "too_many_goals"


def test_goal_save_upserts_by_name_instead_of_duplicating(env):
    first = env.post("/api/goals/learner-110", json={"name": "Car", "target_amount": 10000, "years": 3, "assumed_return_pct": 2})
    assert first.status_code == 201
    gid = first.json()["id"]
    assert len(env.get("/api/goals/learner-110").json()) == 1

    updated = env.post("/api/goals/learner-110", json={"name": "Car", "target_amount": 15000, "years": 4, "assumed_return_pct": 5})
    assert updated.status_code == 200
    assert updated.json()["id"] == gid  # same row updated, not a new one
    rows = env.get("/api/goals/learner-110").json()
    assert len(rows) == 1
    assert rows[0]["goal"]["target_amount"] == 15000
    assert rows[0]["goal"]["assumed_return_pct"] == 5

    # name matching is case- and whitespace-insensitive
    again = env.post("/api/goals/learner-110", json={"name": " car ", "target_amount": 20000, "years": 5})
    assert again.status_code == 200 and again.json()["id"] == gid
    assert len(env.get("/api/goals/learner-110").json()) == 1

    # a genuinely different name still creates a new row
    other = env.post("/api/goals/learner-110", json={"name": "House", "target_amount": 30000, "years": 6})
    assert other.status_code == 201
    assert len(env.get("/api/goals/learner-110").json()) == 2


def test_news_live_sample_and_summary(env):
    r = env.get("/api/news?q=Apple earnings&summarize=true").json()
    assert r["items"][0]["title"] == "Rates hold steady" and r["summary"] and r["notice"] is None
    env.news.status = "sample"
    s = env.get("/api/news?q=rates&summarize=true").json()
    assert s["data_status"] == "sample" and s["notice"] and s["summary"] is None


def test_news_query_is_redacted_before_search(env):
    env.get("/api/news?q=news about jane.doe@example.com and Tesla")
    assert "jane.doe@example.com" not in env.news.calls[-1]


def test_conversation_restore_and_delete(env):
    profile_id = "learner-105"
    sid = SessionRepository(env.svc.db).get_or_create(profile_id, "c1")
    MessageRepository(env.svc.db).add(sid, "user", "What is an ETF?")
    msgs = env.get(f"/api/conversations/{profile_id}/c1").json()["messages"]
    assert msgs and msgs[0]["content"] == "What is an ETF?"
    assert env.delete(f"/api/conversations/{profile_id}/c1").status_code == 204
    assert env.get(f"/api/conversations/{profile_id}/c1").json()["messages"] == []
    assert env.get("/api/conversations/learner-105/never-used").json()["messages"] == []
    assert env.get("/api/conversations/nobody-here/c1").status_code == 404


def test_profile_update(env):
    r = env.put("/api/users/learner-105/profile", json={"risk_tolerance": "conservative", "horizon_years": 7})
    assert r.status_code == 200 and r.json()["risk_tolerance"] == "conservative" and r.json()["horizon_years"] == 7
    assert env.put("/api/users/nobody-here/profile", json={"horizon_years": 7}).status_code == 404
    assert env.put("/api/users/learner-105/profile", json={"risk_tolerance": "yolo"}).status_code == 422


def test_rate_limit_returns_429_with_retry_after_per_client(env):
    env.svc.limiters["read"] = SlidingWindowLimiter(2)
    assert env.get("/api/market/status").status_code == 200
    assert env.get("/api/market/status").status_code == 200
    r = env.get("/api/market/status")
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1 and r.json()["code"] == "too_many_requests"


def test_sliding_window_expires():
    now = [0.0]
    lim = SlidingWindowLimiter(1, window_seconds=10, clock=lambda: now[0])
    assert lim.check("a")[0] and not lim.check("a")[0] and lim.check("b")[0]
    now[0] = 11
    assert lim.check("a")[0]


def test_openapi_documents_new_routes(env):
    paths = env.get("/openapi.json").json()["paths"]
    for p in ("/api/portfolio/{user_id}/analysis", "/api/market/quotes", "/api/goals/project", "/api/news",
              "/api/conversations/{user_id}/{conversation_id}"):
        assert p in paths


def test_limiter_prunes_idle_keys_and_validates_size():
    now = [0.0]
    lim = SlidingWindowLimiter(1, window_seconds=10, max_keys=2, clock=lambda: now[0])
    for key in ("a", "b"):
        lim.check(key)
    now[0] = 20
    lim.check("c"), lim.check("d")
    assert len(lim._hits) <= 3
    with pytest.raises(ValueError):
        SlidingWindowLimiter(0)


def test_sanitize_rejects_bad_text_and_fails_closed(env, monkeypatch):
    import asyncio

    from src.web_app.errors import ApiError

    with pytest.raises(ApiError) as bad:
        asyncio.run(env.svc.sanitize("   "))
    assert bad.value.status_code == 422

    def boom(_):
        raise RuntimeError("nlp down")
    monkeypatch.setattr("src.web_app.api.services.redact_text", boom)
    with pytest.raises(ApiError) as closed:
        asyncio.run(env.svc.sanitize("hello"))
    assert closed.value.status_code == 503 and closed.value.code == "privacy_unavailable"
