import pytest
from fastapi.testclient import TestClient

from src.core.freshness import build_qa_cache_notice
from src.core.models import AgentName, AgentResult
from src.data.db import Database
from src.rag.knowledge_base import KnowledgeBase
from src.rag.qa_cache import QACache
from src.rag.vector_store import SearchResult
from src.web_app.api import main
from tests.unit._agent_fakes import FakeAgent, FakeLLM, FixedStore, make_agent, make_workflow
from tests.unit.test_qa_cache import VecEmbedder


# ------------------------------------------------------------------------------ notice wording
def test_notice_wording():
    exact = build_qa_cache_notice("exact", 5)
    assert "same question" in exact and "only moments old" in exact and "not regenerated" in exact
    sem = build_qa_cache_notice("semantic", 7200, "What is an ETF?")
    assert 'similar question ("What is an ETF?")' in sem and "about 2 hours old" in sem and "your exact wording" in sem
    assert "..." in build_qa_cache_notice("semantic", 60, "x" * 300) and len(build_qa_cache_notice("semantic", 60, "x" * 300)) < 300
    assert "similar question (" in build_qa_cache_notice("semantic", 60) and '("' not in build_qa_cache_notice("semantic", 60)


# ------------------------------------------------------------------------- knowledge-base hook
def test_knowledge_base_notifies_on_ingest_and_reset_and_survives_bad_callbacks():
    class S:
        name = "s"

        def ingest(self, chunks, reset=False):
            return len(chunks)

        def reset(self):
            pass

    kb = KnowledgeBase(S())
    seen = []
    kb.on_change.append(lambda: (_ for _ in ()).throw(RuntimeError("bad callback")))
    kb.on_change.append(lambda: seen.append(1))
    kb.ingest([object()])
    kb.reset()
    assert seen == [1, 1]


def test_api_wires_kb_changes_to_the_cache():
    assert main.qa_cache is not None and main.qa_cache.bump_version in main.store.on_change
    assert main.finance_qa.cache is main.qa_cache


# ---------------------------------------------------------------------------------- API end to end
@pytest.fixture
def api(tmp_path, monkeypatch):
    db = Database(tmp_path / "api.db")
    db.init_schema()
    cache = QACache(db)
    llm = FakeLLM()
    store = FixedStore([SearchResult("ETF passage", "ETF", {"source_name": "Investopedia", "source_url": "https://ex.com/etf", "category": "funds"}, 0.7)])
    agent = make_agent(store, llm, cache=cache, embed=VecEmbedder({}))
    monkeypatch.setattr(main, "workflow", make_workflow({"finance_qa": agent}))
    monkeypatch.setattr(main, "qa_cache", cache)
    client = TestClient(main.app)
    client.llm, client.cache, client.store, client.agent = llm, cache, store, agent
    return client


def chat(client, question, **extra):
    return client.post("/api/chat", json={"user_id": "learner-204", "question": question, **extra}).json()


def test_api_marks_cached_answers_and_exposes_the_notice(api):
    first = chat(api, "What is an ETF and how does it work?")
    assert first["cache_hit"] is None and first["data_status"] == "live" and first["notice"] is None
    second = chat(api, "What is an ETF and how does it work?")
    assert second["cache_hit"] == "exact" and second["data_status"] == "cached"
    assert "same question" in second["notice"] and second["answer"] == first["answer"] and api.llm.calls == 1
    assert second["sources"][0]["url"] == "https://ex.com/etf"  # sources survive the cache round trip
    assert second["cached_question"] is None


def test_api_reports_the_matched_question_for_semantic_hits(api):
    api.cache.threshold = 0.5
    api.cache.store("What is an ETF and how does it work?", "Saved answer.", [SearchResult("ETF passage", "ETF", {}, 0.7)],
                    embed=lambda t: [1.0, 0, 0, 0, 0, 0, 0, 0])
    api.agent.embed = lambda t: [1.0, 0.1, 0, 0, 0, 0, 0, 0]
    body = chat(api, "Explain what an ETF is and how it works")
    assert body["cache_hit"] == "semantic" and body["cached_question"] == "What is an ETF and how does it work?"
    assert "similar question" in body["notice"]


def test_api_does_not_cache_questions_containing_pii(api):
    q = "I'm John Smith, email john.smith@example.com. What is an ETF and how does it work?"
    a, b = chat(api, q), chat(api, q)
    assert a["pii_redactions"] and b["cache_hit"] is None and b["notice"] is None
    assert api.llm.calls == 2 and api.cache.stats()["entries"] == 0


def test_api_category_scopes_the_cache(api):
    chat(api, "What is an ETF and how does it work?", category="funds")
    assert chat(api, "What is an ETF and how does it work?", category="tax")["cache_hit"] is None


def test_api_stats_report_cache_effectiveness(api):
    chat(api, "What is an ETF and how does it work?")
    chat(api, "What is an ETF and how does it work?")
    stats = api.get("/api/stats").json()["qa_cache"]
    assert stats["entries"] == 1 and stats["exact_hits"] == 1 and stats["stores"] == 1 and stats["hit_rate"] == 0.5


def test_api_remembers_conversation_turns_and_sends_them_to_the_agent(api):
    chat(api, "What is an ETF and how does it work?", conversation_id="c-1")
    api.llm.messages.clear()
    chat(api, "And what about bonds?", conversation_id="c-1")
    sent = api.llm.messages[0]
    assert any(getattr(m, "content", "") == "What is an ETF and how does it work?" for m in sent)  # earlier turn included
    chat(api, "And what about bonds?", conversation_id="c-2")  # a different conversation has no history
    assert not any(getattr(m, "content", "") == "What is an ETF and how does it work?" for m in api.llm.messages[1])


def test_api_does_not_remember_refusals(api, monkeypatch):
    api.store.results = [SearchResult("Unrelated", "X", {}, 0.01)]
    chat(api, "What is the best pizza recipe in Chicago?", conversation_id="c-3")
    assert main.workflow.memory.history(main.workflow.memory.key("learner-204", "c-3")) == []


@pytest.mark.parametrize("code,status", [("knowledge_base_empty", 409), ("llm_unavailable", 503), ("rate_limited", 503),
                                         ("guardrail_rejected", 422), ("internal_error", 500), ("something_new", 500)])
def test_api_maps_agent_error_codes_to_http_status(monkeypatch, code, status):
    monkeypatch.setattr(main, "workflow", make_workflow([FakeAgent("finance_qa", error="Readable message", error_code=code)]))
    r = TestClient(main.app).post("/api/chat", json={"user_id": "learner-204", "question": "What is an ETF?"})
    assert r.status_code == status and r.json()["detail"] == "Readable message"
