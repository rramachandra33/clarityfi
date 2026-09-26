import asyncio

import pytest

from src.agents.base import INTERNAL_ERROR_MESSAGE, AgentRequest, BaseAgent
from src.core import tracing
from src.core.errors import GuardrailError, LLMError, ProviderUnavailableError
from src.core.freshness import DataStatus
from src.core.models import AgentName, AgentResult
from src.workflow.memory import ConversationMemory


class Echo(BaseAgent):
    name = AgentName.market
    description = "test agent"

    def __init__(self, behavior="ok"):
        self.behavior, self.requests = behavior, []

    async def _execute(self, request):
        self.requests.append(request)
        if self.behavior == "clarity":
            raise ProviderUnavailableError()
        if self.behavior == "llm":
            raise LLMError("model is down")
        if self.behavior == "boom":
            raise RuntimeError("secret internal detail: password=hunter2")
        if self.behavior == "cancel":
            raise asyncio.CancelledError()
        return AgentResult(agent=AgentName.tax, content=f"echo: {request.question}", notice="cached data", data_status=DataStatus.cached)


def req(**kw):
    return AgentRequest(question=kw.pop("question", "hello"), **kw)


# ------------------------------------------------------------------------------------ BaseAgent
def test_successful_run_sets_agent_name_and_latency():
    result = Echo().run(req())
    assert result.ok and result.content == "echo: hello" and result.agent is AgentName.market  # base overrides the name
    assert result.data["latency_ms"] >= 0 and result.notice == "cached data" and result.data_status is DataStatus.cached


@pytest.mark.parametrize("behavior,code,fragment", [
    ("clarity", "provider_unavailable", "temporarily unavailable"), ("llm", "llm_unavailable", "model is down"),
])
def test_expected_errors_become_results_with_codes(behavior, code, fragment):
    result = Echo(behavior).run(req())
    assert not result.ok and result.error_code == code and fragment in result.error and result.content == ""


def test_unexpected_errors_are_contained_and_never_leak_details(caplog):
    with caplog.at_level("ERROR"):
        result = Echo("boom").run(req())
    assert result.error == INTERNAL_ERROR_MESSAGE and result.error_code == "internal_error"
    assert "hunter2" not in result.error and "hunter2" not in str(result.data)
    assert any("Unexpected error in market agent" in m for m in caplog.messages)


def test_cancellation_is_not_swallowed():
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(Echo("cancel").arun(req()))


def test_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseAgent()


def test_request_from_state_defaults_and_conversion():
    r = AgentRequest.from_state({"question": "hi"})
    assert (r.user_id, r.session_id, r.top_k, r.cacheable, r.history, r.context, r.category) == ("anonymous", "default", 3, True, [], {}, None)
    full = AgentRequest.from_state({"question": 5, "user_id": "u", "session_id": "s", "category": "tax", "top_k": "7",
                                    "history": [{"role": "user", "content": "x"}], "cacheable": False, "context": {"a": 1}})
    assert (full.question, full.user_id, full.top_k, full.cacheable, full.context, full.category) == ("5", "u", 7, False, {"a": 1}, "tax")
    with pytest.raises(KeyError):
        AgentRequest.from_state({})


def test_as_node_is_a_langgraph_compatible_async_function():
    agent = Echo()
    node = agent.as_node()
    assert node.__name__ == "market_node" and asyncio.iscoroutinefunction(node)
    update = asyncio.run(node({"question": "from state", "user_id": "u-1", "category": "tax"}))
    assert set(update) == {"agent_results"} and update["agent_results"][0].content == "echo: from state"
    assert agent.requests[0].user_id == "u-1" and agent.requests[0].category == "tax"


def test_as_node_works_inside_a_real_langgraph_graph():
    from operator import add
    from typing import Annotated, TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict, total=False):
        question: str
        agent_results: Annotated[list, add]

    graph = StateGraph(State)
    graph.add_node("market", Echo().as_node())
    graph.add_edge(START, "market")
    graph.add_edge("market", END)
    out = asyncio.run(graph.compile().ainvoke({"question": "via graph", "agent_results": []}))
    assert out["agent_results"][0].content == "echo: via graph"


def test_tracing_span_receives_input_output_and_metadata(monkeypatch):
    seen = {}

    class Span:
        def update(self, **kw):
            seen.update(kw)

    import contextlib

    @contextlib.contextmanager
    def fake_span(name, as_type="span", **attrs):
        seen["name"], seen["as_type"], seen["input"] = name, as_type, attrs.get("input")
        yield Span()

    @contextlib.contextmanager
    def fake_ctx(**kw):
        seen["ctx"] = kw
        yield

    from src.agents import base as base_mod
    monkeypatch.setattr(base_mod, "trace_span", fake_span)
    monkeypatch.setattr(base_mod, "trace_context", fake_ctx)
    Echo().run(req(question="What is an ETF?", user_id="a@b.com", session_id="conv-9", category="funds"))
    assert seen["name"] == "agent.market" and seen["as_type"] == "agent"
    assert seen["input"] == {"question": "What is an ETF?", "category": "funds"}
    assert seen["output"]["error_code"] is None and seen["metadata"]["data_status"] == "cached"
    assert seen["ctx"]["tags"] == ["agent:market"]


# ------------------------------------------------------------------------------- tracing helpers
def test_pseudonym_is_stable_non_reversible_and_hides_emails():
    a = tracing.pseudonym("Jane.Doe@Example.com", "u_")
    assert a == tracing.pseudonym("  jane.doe@example.com ", "u_") and a.startswith("u_") and "jane" not in a and "@" not in a
    assert tracing.pseudonym("") is None and tracing.pseudonym(None) is None
    assert tracing.pseudonym("x") != tracing.pseudonym("y")


def test_trace_context_and_callbacks_are_noops_without_keys():
    s = tracing.settings
    saved = (s.langfuse_public_key, s.langfuse_secret_key)
    object.__setattr__(s, "langfuse_public_key", "")
    object.__setattr__(s, "langfuse_secret_key", "")
    tracing.get_langfuse.cache_clear()
    try:
        with tracing.trace_context(user_id="u", session_id="s", tags=["x"]):
            pass
        assert tracing.langfuse_callbacks() == []
    finally:
        object.__setattr__(s, "langfuse_public_key", saved[0])
        object.__setattr__(s, "langfuse_secret_key", saved[1])
        tracing.get_langfuse.cache_clear()


def test_langfuse_callbacks_fail_safe_when_handler_import_breaks(monkeypatch):
    monkeypatch.setattr(tracing, "get_langfuse", lambda: object())
    import builtins
    real_import = builtins.__import__

    def broken(name, *a, **k):
        if name.startswith("langfuse.langchain"):
            raise ModuleNotFoundError("langchain missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", broken)
    assert tracing.langfuse_callbacks() == []


def test_trace_context_survives_propagation_errors(monkeypatch):
    import langfuse

    monkeypatch.setattr(tracing, "get_langfuse", lambda: object())
    monkeypatch.setattr(langfuse, "propagate_attributes", lambda **kw: (_ for _ in ()).throw(RuntimeError("otel broke")), raising=False)
    ran = []
    with tracing.trace_context(user_id="u"):
        ran.append(1)
    assert ran == [1]


# ------------------------------------------------------------------------------------- memory
def test_memory_records_and_returns_history_capped_per_conversation():
    m = ConversationMemory(max_messages=4)
    key = m.key(" Learner-204 ", "finance-qa")
    assert key == "learner-204:finance-qa" and m.history(key) == []
    for i in range(3):
        m.record_exchange(key, f"q{i}", f"a{i}")
    h = m.history(key)
    assert [t["content"] for t in h] == ["q1", "a1", "q2", "a2"] and h[0]["role"] == "user" and h[1]["role"] == "assistant"


def test_memory_ignores_invalid_roles_and_blank_content_and_returns_copies():
    m = ConversationMemory()
    m.append("k", "system", "nope")
    m.append("k", "user", "   ")
    m.append("k", "user", "hello")
    snapshot = m.history("k")
    snapshot.append({"role": "user", "content": "tampered"})
    assert m.history("k") == [{"role": "user", "content": "hello"}]


def test_memory_isolates_conversations_and_bounds_total_size():
    m = ConversationMemory(max_conversations=3)
    for i in range(5):
        m.append(f"k{i}", "user", f"m{i}")
    assert m.history("k0") == [] and m.history("k1") == [] and m.history("k4") == [{"role": "user", "content": "m4"}]
    m.append("k2", "user", "again")  # touching k2 makes it most recent
    m.append("k5", "user", "new")
    assert m.history("k2") and m.history("k3") == []


def test_memory_clear():
    m = ConversationMemory()
    m.append("a", "user", "x")
    m.append("b", "user", "y")
    m.clear("a")
    assert m.history("a") == [] and m.history("b")
    m.clear()
    assert m.history("b") == []


# ------------------------------------------------------------------------- persistent event loop
def test_sync_run_reuses_one_event_loop_so_cached_async_clients_stay_valid():
    """Regression (found live): asyncio.run() per call closed the loop, so LangChain's cached HTTP client
    failed on the second call with 'Event loop is closed'."""
    loops = []

    class Probe(BaseAgent):
        name = AgentName.tax

        async def _execute(self, request):
            loops.append(asyncio.get_running_loop())
            await asyncio.sleep(0)
            return AgentResult(agent=self.name, content="ok")

    agent = Probe()
    for _ in range(3):
        assert agent.run(req()).ok
    assert len({id(l) for l in loops}) == 1 and not loops[0].is_closed() and loops[0].is_running()


def test_sync_run_is_thread_safe():
    import threading

    agent = Echo()
    results = []

    def worker(i):
        results.append(agent.run(req(question=f"q{i}")).content)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == sorted(f"echo: q{i}" for i in range(12))


def test_sync_run_refuses_to_run_inside_an_event_loop_and_does_not_leak_the_coroutine():
    async def inner():
        with pytest.raises(RuntimeError, match="use `await agent.arun"):
            Echo().run(req())

    asyncio.run(inner())


def test_entering_and_leaving_are_logged_at_info(caplog):
    with caplog.at_level("INFO"):
        Echo().run(req())
    messages = caplog.messages
    assert any("Entering market agent" in m for m in messages)
    assert any(m.startswith("Leaving market agent after") for m in messages)
    assert messages.index(next(m for m in messages if "Entering market agent" in m)) < \
        messages.index(next(m for m in messages if m.startswith("Leaving market agent after")))
