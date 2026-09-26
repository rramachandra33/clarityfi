"""Doubles for agent tests: a LangChain-like chat model and simple retrieval stores."""
from types import SimpleNamespace

from src.agents.base import AgentRequest
from src.agents.finance_qa import FinanceQAAgent
from src.rag.vector_store import SearchResult


class FakeLLM:
    """Async chat model with the `ainvoke(messages, config=...)` shape used by ainvoke_chat."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error
        self.calls, self.messages, self.configs = 0, [], []

    async def ainvoke(self, messages, config=None, **kwargs):
        self.calls += 1
        self.messages.append(list(messages))
        self.configs.append(config)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.reply if self.reply is not None else f"Generated answer #{self.calls}.")


def hit(title="T", score=0.7, **meta):
    return SearchResult(text=f"{title} passage", title=title, metadata=meta, score=score)


class FixedStore:
    def __init__(self, results, count=10):
        self.results, self._count, self.calls = results, count, []

    def search(self, query, top_k, *, category=None, mode="hybrid"):
        self.calls.append({"query": query, "top_k": top_k, "category": category})
        return list(self.results)

    def count(self):
        return self._count


def make_agent(store, llm=None, *, cache=None, embed=None, threshold=0.25):
    llm = llm or FakeLLM()
    agent = FinanceQAAgent(store, cache=cache, embed=embed, score_threshold=threshold,
                           llm_factory=lambda: llm, callbacks_factory=lambda: ["cb"])
    agent.llm = llm
    return agent


def ask(agent, question, **kw):
    return agent.run(AgentRequest(question=question, **kw))


# ---------------------------------------------------------------------------- market-agent doubles
from datetime import date, timedelta  # noqa: E402

from src.core.freshness import DataStatus  # noqa: E402
from src.core.models import PricePoint, Quote  # noqa: E402
from src.core.market_calc import compute_trend  # noqa: E402
from src.data.market_client import History, QuoteBatch  # noqa: E402
from src.utils.validators import normalize_ticker  # noqa: E402


def quote(symbol, price, *, change=0.5, status=DataStatus.live, source="alphavantage", age=600):
    q = Quote(symbol=symbol, price=price, change_pct=change, previous_close=None if change is None else round(price / (1 + change / 100), 2),
              trading_day=date(2026, 9, 18), source=source)
    return q if status is DataStatus.live else q.with_status(status, age)


class FakeMarket:
    """MarketService double: `quotes` maps symbol -> Quote; symbols not in it are 'not found'."""

    def __init__(self, quotes, *, trend_status=DataStatus.live, trend_fails=False):
        self.quotes, self.trend_status, self.trend_fails = quotes, trend_status, trend_fails
        self.quote_calls, self.trend_calls = [], []

    def get_quotes(self, symbols):
        self.quote_calls.append(list(symbols))
        batch = QuoteBatch()
        for raw in symbols:
            sym = normalize_ticker(raw)
            if sym in self.quotes:
                batch.quotes[sym] = self.quotes[sym]
            else:
                batch.errors[sym] = "Ticker was not found"
        return batch

    def status(self):
        return [{"provider": "alphavantage", "sample": False, "quota_remaining": 20, "ok": True}]

    def get_trend(self, symbol, days=None):
        self.trend_calls.append(symbol)
        if self.trend_fails:
            raise RuntimeError("history unavailable")
        start = date(2026, 3, 1)
        points = [PricePoint(day=start + timedelta(days=i), close=100 + i * 0.4) for i in range(120)]
        history = History(symbol, points, "alphavantage")
        trend = compute_trend(symbol, points)
        if self.trend_status is not DataStatus.live:
            history = History(symbol, points, "alphavantage", status=self.trend_status, age_seconds=7200)
            trend.data_status, trend.notice = history.status, history.notice
            trend.notes.insert(0, history.notice)
        return trend, history


# ----------------------------------------------------------------------------------- workflow doubles
from src.agents.base import BaseAgent  # noqa: E402
from src.core.models import AgentName, AgentResult, Citation  # noqa: E402
from src.workflow.graph import ChatWorkflow  # noqa: E402
from src.workflow.memory import ConversationMemory  # noqa: E402
from src.workflow.router import Router  # noqa: E402


class FakeAgent(BaseAgent):
    """A scripted agent: returns `content` (or raises/sleeps) and records every request it receives."""

    def __init__(self, name, content=None, *, notices=None, citations=None, data=None, delay=0.0, error=None,
                 error_code="llm_unavailable", raises=None, status=DataStatus.live, cache_hit=None):
        self.name = AgentName(name)
        self.content = content if content is not None else f"{name} answer."
        self.notices, self.citations, self.data = notices or [], citations or [], data or {}
        self.delay, self.error, self.error_code, self.raises = delay, error, error_code, raises
        self.status, self.cache_hit = status, cache_hit
        self.requests = []

    async def _execute(self, request):
        import asyncio

        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises:
            raise self.raises
        if self.error:
            return AgentResult(agent=self.name, error=self.error, error_code=self.error_code)
        content = self.content
        if self.notices:
            content += "\n\nData note:\n" + "\n".join(f"- {n}" for n in self.notices)
        return AgentResult(agent=self.name, content=content, citations=self.citations, data=dict(self.data),
                           notice=" ".join(self.notices) or None, data_status=self.status, cache_hit=self.cache_hit)


def make_workflow(agents, *, memory=None, router=None, timeout=5.0):
    """A workflow over fake agents with rules-only routing (deterministic, no LLM, no network)."""
    if isinstance(agents, list):
        agents = {a.name.value: a for a in agents}
    return ChatWorkflow(agents, router or Router(use_llm=False), memory or ConversationMemory(), agent_timeout=timeout)


def cite(index=1, title="ETF", source="Investopedia", url="https://ex.com/etf"):
    return Citation(index=index, title=title, source=source, url=url, category="funds", excerpt="excerpt", score=0.7)
