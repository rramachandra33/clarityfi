"""End-to-end stack: the real FastAPI app, routers, workflow, PII filter and SQLite, with deterministic stand-ins for the
things that cost money or need the network (LLM, prices, news, knowledge base). Nothing here calls an external service."""
import hashlib
import math
from datetime import date

import pytest
from fastapi.testclient import TestClient

from src.agents.finance_qa import FinanceQAAgent
from src.agents.goal_planning import GoalParams, GoalPlanningAgent
from src.agents.market import MarketAgent
from src.agents.news import NewsAgent
from src.agents.portfolio import PortfolioAgent
from src.agents.tax import TaxEducationAgent
from src.core.audio import AudioService
from src.core.models import NewsItem
from src.data.db import Database
from src.data.seed import seed
from src.rag.qa_cache import QACache
from src.web_app.api import main
from src.web_app.api.services import ApiServices, default_limiters
from src.web_app.auth_service import AuthService
from src.workflow.graph import ChatWorkflow
from src.workflow.memory import SqliteConversationMemory
from src.workflow.router import Router
from tests.unit._agent_fakes import FakeLLM, FakeMarket, FixedStore, hit, quote

PRICES = {"VTI": 300.0, "VXUS": 60.0, "BND": 75.0, "AAPL": 200.0, "TSLA": 250.0, "MSFT": 400.0, "SPY": 500.0, "QQQ": 450.0,
          "VOO": 480.0, "AGG": 100.0, "BTC-USD": 60000.0, "NVDA": 900.0, "VNQ": 90.0, "VGT": 550.0, "SCHD": 80.0}


def embed(text: str) -> list[float]:
    """Deterministic pseudo-embedding: identical text -> identical vector, different text -> near-orthogonal."""
    digest = hashlib.sha256(text.lower().encode()).digest()
    vec = [b - 127.5 for b in digest[:16]]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


class Headlines:
    def __init__(self):
        self.status, self.calls = "live", []

    def search(self, query, limit=6):
        self.calls.append(query)
        notice = None if self.status == "live" else "These are SAMPLE headlines, not real news."
        items = [NewsItem(title="Rates hold steady as inflation cools", url="https://example.com/rates", source="Wire", published="today")]
        return items, {"source": "serpapi", "data_status": self.status, "notice": notice, "stale": False, "cache_hit": False}


class Stack:
    """Handles the tests need: client, db, fakes, and the LLM double. The client is signed in through the real auth flow."""

    current_user: str | None = None

    def as_user(self, user_id: str, password: str | None = None) -> None:
        """Sign in as `user_id` (demo users use the demo password; other ids are registered on first use)."""
        if self.current_user and self.current_user.lower() == user_id.lower():
            return
        from src.core.config import settings

        self.client.cookies.clear()
        if user_id.startswith("learner-1") and self.db_has_password(user_id):
            r = self.client.post("/api/auth/login", json={"user_id": user_id, "password": password or settings.demo_password})
        else:
            r = self.client.post("/api/auth/register", json={"user_id": user_id, "password": password or "a-decent-password"})
        assert r.status_code in (200, 201), r.text
        self.current_user = r.json()["user_id"]

    def db_has_password(self, user_id: str) -> bool:
        from src.data.repositories import AuthRepository

        return bool(AuthRepository(self.db).get_hash(user_id))


@pytest.fixture
def stack(tmp_path, monkeypatch):
    s = Stack()
    s.db = Database(tmp_path / "e2e.db")
    s.db.init_schema()
    seed(s.db)
    s.market = FakeMarket({t: quote(t, p) for t, p in PRICES.items()})
    s.news = Headlines()
    s.llm = FakeLLM("An ETF is a basket of investments that trades like a stock. [1] Educational only.")
    kb = FixedStore([hit("What is an ETF?", 0.82, category="funds", source_name="Investopedia", source_url="https://example.com/etf")])
    s.kb = kb
    cache = QACache(s.db)
    llm_factory = lambda: s.llm  # noqa: E731
    no_cb = lambda: []  # noqa: E731

    async def extract(question):  # stands in for the LLM extraction; values are then verified against the user's own words
        return GoalParams(target_amount=50000, years=8, current_savings=5000, monthly_contribution=300)

    agents = {
        "finance_qa": FinanceQAAgent(kb, cache=cache, embed=embed, llm_factory=llm_factory, callbacks_factory=no_cb),
        "tax": TaxEducationAgent(kb, cache=cache, embed=embed, llm_factory=llm_factory, callbacks_factory=no_cb),
        "portfolio": PortfolioAgent(s.db, s.market, llm_factory=llm_factory, callbacks_factory=no_cb),
        "market": MarketAgent(s.market, llm_factory=llm_factory, callbacks_factory=no_cb),
        "goal_planning": GoalPlanningAgent(s.db, llm_factory=llm_factory, callbacks_factory=no_cb, extractor=extract, today=lambda: date(2026, 9, 21)),
        "news": NewsAgent(s.news, llm_factory=llm_factory, callbacks_factory=no_cb),
    }
    memory = SqliteConversationMemory(s.db)
    workflow = ChatWorkflow(agents, Router(use_llm=False), memory)
    services = ApiServices(db=s.db, market=s.market, news=s.news, portfolio_agent=agents["portfolio"], goal_agent=agents["goal_planning"],
                           news_agent=agents["news"], memory=memory, audio=AudioService(None), limiters=default_limiters())
    monkeypatch.setattr(main, "workflow", workflow)
    monkeypatch.setattr(main, "api_services", services)
    auth_service = AuthService(s.db)
    auth_service.ensure_demo_passwords()
    monkeypatch.setattr(main.app.state, "auth", auth_service)
    main.app.dependency_overrides[main.get_services] = lambda: services
    main.app.dependency_overrides[main.get_db] = lambda: s.db
    s.services = services
    with TestClient(main.app, headers={"X-Clarity-Client": "web"}) as client:
        s.client = client
        s.as_user("learner-105")
        yield s
    main.app.dependency_overrides.clear()
