import pytest
from fastapi.testclient import TestClient

from src.core import tracing


@pytest.fixture
def no_keys():
    s = tracing.settings
    saved = (s.langfuse_public_key, s.langfuse_secret_key)
    object.__setattr__(s, "langfuse_public_key", "")
    object.__setattr__(s, "langfuse_secret_key", "")
    tracing.get_langfuse.cache_clear()
    yield
    object.__setattr__(s, "langfuse_public_key", saved[0])
    object.__setattr__(s, "langfuse_secret_key", saved[1])
    tracing.get_langfuse.cache_clear()


def test_tracing_is_noop_without_keys(no_keys):
    assert tracing.tracing_enabled() is False
    assert tracing.get_langfuse() is None
    assert tracing.langfuse_callbacks() == []
    with tracing.trace_span("x", input="anything") as span:
        assert span is None
    tracing.flush_traces()  # must not raise


def test_flush_swallows_client_errors(monkeypatch):
    class Bad:
        def flush(self):
            raise RuntimeError("network down")

    monkeypatch.setattr(tracing, "get_langfuse", lambda: Bad())
    tracing.flush_traces()  # observability must never break the app


def test_trace_span_uses_client_when_enabled(monkeypatch):
    events = []

    class Span:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *exc):
            events.append("exit")

    class Client:
        def start_as_current_observation(self, **kw):
            events.append(kw["name"])
            return Span()

    monkeypatch.setattr(tracing, "get_langfuse", lambda: Client())
    with tracing.trace_span("agent.run"):
        pass
    assert events == ["agent.run", "enter", "exit"]


def test_api_adds_request_id_header_and_honors_incoming():
    from src.web_app.api.main import app

    client = TestClient(app)
    generated = client.get("/api/stats")
    assert generated.status_code == 200
    assert len(generated.headers["x-request-id"]) == 12
    echoed = client.get("/api/stats", headers={"x-request-id": "trace-me-1"})
    assert echoed.headers["x-request-id"] == "trace-me-1"
