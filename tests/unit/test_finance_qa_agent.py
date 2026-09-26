import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.finance_qa import NO_ANSWER, FinanceQAAgent, is_context_dependent, looks_like_followup
from src.core.disclaimers import PLAIN_DISCLAIMER
from src.core.errors import LLMError, RateLimitedError
from src.core.freshness import DataStatus
from src.core.models import AgentName
from src.data.db import Database
from src.rag.qa_cache import QACache
from tests.unit._agent_fakes import FakeLLM, FixedStore, ask, hit, make_agent
from tests.unit.test_qa_cache import VecEmbedder, at_cosine, unit


# ---------------------------------------------------------------------------------- grounding
def test_relevant_results_produce_a_cited_grounded_answer():
    agent = make_agent(FixedStore([hit("ETF", 0.7, source_name="Investopedia", source_url="https://ex.com/etf", category="funds")]))
    result = ask(agent, "What is an ETF?")
    assert result.ok and result.agent is AgentName.finance_qa and result.content.startswith("Generated answer #1.")
    c = result.citations[0]
    assert (c.index, c.title, c.source, c.url, c.category) == (1, "ETF", "Investopedia", "https://ex.com/etf", "funds")
    prompt = agent.llm.messages[0][-1].content
    assert "[1] Title: ETF" in prompt and "Source: Investopedia" in prompt and "What is an ETF?" in prompt
    assert isinstance(agent.llm.messages[0][0], SystemMessage) and "only from the supplied knowledge-base context" in agent.llm.messages[0][0].content
    assert result.data["sources_used"] == 1 and result.data["latency_ms"] >= 0 and result.cache_hit is None
    assert result.data_status is DataStatus.live and result.notice is None


def test_langfuse_callbacks_are_passed_to_the_llm_call():
    agent = make_agent(FixedStore([hit("ETF", 0.7)]))
    ask(agent, "What is an ETF?")
    assert agent.llm.configs == [{"callbacks": ["cb"]}]


def test_low_relevance_is_refused_without_calling_the_llm():
    agent = make_agent(FixedStore([hit("Unrelated", 0.08), hit("Also", 0.2)]))
    result = ask(agent, "best pizza in Chicago")
    assert result.content == NO_ANSWER and result.citations == [] and agent.llm.calls == 0
    assert result.data["refusal"] is True and result.data["reason"] == "low_relevance" and result.data["top_score"] == 0.2


def test_only_relevant_passages_reach_the_prompt():
    agent = make_agent(FixedStore([hit("Good", 0.6), hit("Bad", 0.1), hit("OK", 0.3)]))
    result = ask(agent, "What is an ETF?")
    assert [c.title for c in result.citations] == ["Good", "OK"] and "Bad" not in agent.llm.messages[0][-1].content


def test_threshold_defaults_from_config():
    assert FinanceQAAgent(FixedStore([])).score_threshold == 0.25


def test_empty_knowledge_base_is_a_contained_error_but_no_match_is_a_refusal():
    empty = ask(make_agent(FixedStore([], count=0)), "What is an ETF?")
    assert not empty.ok and empty.error_code == "knowledge_base_empty" and "empty" in empty.error
    populated = ask(make_agent(FixedStore([], count=200)), "anything at all", category="no_such")
    assert populated.ok and populated.content == NO_ANSWER and populated.data["reason"] == "no_match"


def test_category_and_top_k_are_forwarded_to_retrieval():
    store = FixedStore([hit("Tax", 0.7)])
    ask(make_agent(store), "How are dividends taxed?", category="tax", top_k=5)
    assert store.calls[0] == {"query": "How are dividends taxed?", "top_k": 5, "category": "tax"}


# --------------------------------------------------------------------------- disclaimer guarantee
def test_missing_disclaimer_is_appended_and_existing_one_is_not_duplicated():
    plain = ask(make_agent(FixedStore([hit("ETF", 0.7)]), FakeLLM("An ETF is a basket. [1]")), "What is an ETF?")
    assert plain.content.endswith(PLAIN_DISCLAIMER)
    has = ask(make_agent(FixedStore([hit("ETF", 0.7)]), FakeLLM("An ETF is a basket. [1] Educational only.")), "What is an ETF?")
    assert has.content == "An ETF is a basket. [1] Educational only." and PLAIN_DISCLAIMER not in has.content


# --------------------------------------------------------------------------------- llm failures
@pytest.mark.parametrize("error,code", [
    (RateLimitedError(), "rate_limited"), (LLMError("down"), "llm_unavailable"),
    (type("RateLimitError", (Exception,), {"status_code": 429})("x"), "rate_limited"),
    (type("APIConnectionError", (Exception,), {})("x"), "llm_unavailable"),
])
def test_llm_failures_become_error_results_with_codes(error, code):
    result = ask(make_agent(FixedStore([hit("ETF", 0.7)]), FakeLLM(error=error)), "What is an ETF?")
    assert not result.ok and result.error_code == code and result.content == ""


def test_empty_llm_answer_is_an_error_not_a_blank_reply():
    result = ask(make_agent(FixedStore([hit("ETF", 0.7)]), FakeLLM("   ")), "What is an ETF?")
    assert result.error_code == "llm_unavailable"


# ------------------------------------------------------------------------------------- history
def test_history_is_sent_as_alternating_langchain_messages_and_capped():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(12)]
    agent = make_agent(FixedStore([hit("ETF", 0.7)]))
    ask(agent, "What is an ETF?", history=history)
    sent = agent.llm.messages[0]
    assert isinstance(sent[0], SystemMessage) and isinstance(sent[-1], HumanMessage)
    middle = sent[1:-1]
    assert len(middle) == 8 and middle[0].content == "turn 4" and middle[-1].content == "turn 11"
    assert isinstance(middle[0], HumanMessage) and isinstance(middle[1], AIMessage)


def test_unrelated_new_question_is_refused_even_after_an_on_topic_turn():
    """Regression: an earlier topic must not make an off-topic question look relevant."""
    class TopicStore(FixedStore):
        def search(self, query, top_k, *, category=None, mode="hybrid"):
            self.calls.append({"query": query})
            return [hit("Topic", 0.7 if "diversification" in query.lower() else 0.3 if "\n" in query else 0.05)]

    store = TopicStore([])
    agent = make_agent(store)
    history = [{"role": "user", "content": "What is diversification and why does it matter?"},
               {"role": "assistant", "content": "It spreads risk."}]
    result = ask(agent, "What is the best pizza recipe in Chicago?", history=history)
    assert result.content == NO_ANSWER and agent.llm.calls == 0
    assert all("\n" not in c["query"] for c in store.calls)  # never blended with the earlier question


def test_short_followup_borrows_context_only_when_needed():
    class TopicStore(FixedStore):
        def search(self, query, top_k, *, category=None, mode="hybrid"):
            self.calls.append({"query": query})
            return [hit("Topic", 0.7 if "dividend" in query.lower() else 0.05)]

    store = TopicStore([])
    agent = make_agent(store)
    history = [{"role": "user", "content": "How are dividends taxed?"}, {"role": "assistant", "content": "..."}]
    result = ask(agent, "And for bonds?", history=history)
    assert result.ok and result.citations and store.calls[-1]["query"] == "How are dividends taxed?\nAnd for bonds?"
    n = len(store.calls)
    ask(agent, "What is a dividend yield?", history=history)  # self-contained: no extra context search
    assert len(store.calls) == n + 1


def test_short_followup_is_refused_when_neither_alone_nor_with_context_is_relevant():
    agent = make_agent(FixedStore([hit("Topic", 0.08)]))
    history = [{"role": "user", "content": "How are dividends taxed?"}, {"role": "assistant", "content": "..."}]
    assert ask(agent, "pizza?", history=history).content == NO_ANSWER and agent.llm.calls == 0


@pytest.mark.parametrize("question,expected", [
    ("And capital gains?", True), ("why does it matter?", True), ("What about bonds", True), ("Is it risky for me to do that now", True),
    ("What is the best pizza recipe in Chicago?", False), ("Explain how a Roth IRA differs from a 401k", False),
    ("Tell me how to repair the brakes on my old bicycle", False), ("but that seems expensive for a long question here", True),
])
def test_looks_like_followup(question, expected):
    assert looks_like_followup(question) is expected


@pytest.mark.parametrize("question,dependent", [
    ("What is an ETF?", False), ("What is a bond?", False), ("How do bonds work?", False), ("How does compound interest work?", False),
    ("Explain the difference between a Roth IRA and a 401k", False), ("What is dollar-cost averaging?", False),
    ("Why does it matter?", True), ("And for bonds?", True), ("Which is better?", True), ("Tell me more", True),
    ("What about taxes?", True), ("Is it risky?", True), ("Can you explain that again?", True), ("bonds?", True), ("Why?", True),
    ("How does this apply to real estate?", True), ("Okay, and what else should I know?", True), ("Explain those in simple terms", True),
])
def test_is_context_dependent(question, dependent):
    assert is_context_dependent(question) is dependent


# ------------------------------------------------------------------------------------- caching
@pytest.fixture
def clock():
    return [50_000.0]


@pytest.fixture
def cached(tmp_path, clock):
    db = Database(tmp_path / "agent.db")
    db.init_schema()
    cache = QACache(db, exact_ttl=86400, semantic_ttl=86400, threshold=0.92, clock=lambda: clock[0])
    emb = VecEmbedder({"What is an ETF?": unit(1), "Could you explain what an ETF is?": at_cosine(0.96)})
    store = FixedStore([hit("ETF", 0.7, source_name="Investopedia", source_url="https://ex.com/etf", category="funds")])
    agent = make_agent(store, cache=cache, embed=emb)
    return agent, store, cache


def test_second_identical_question_is_served_from_cache_with_a_notice(cached, clock):
    agent, store, cache = cached
    first = ask(agent, "What is an ETF?")
    assert first.cache_hit is None and first.notice is None and agent.llm.calls == 1
    clock[0] += 3 * 3600
    second = ask(agent, "what is an etf", session_id="another-conversation")
    assert agent.llm.calls == 1 and len(store.calls) == 1  # no retrieval, no LLM
    assert second.cache_hit == "exact" and second.data_status is DataStatus.cached and second.content == first.content
    assert second.citations[0].url == "https://ex.com/etf"
    assert "same question" in second.notice and "3 hours old" in second.notice and "not regenerated" in second.notice
    assert second.data["retrieval"] == "cache" and second.data["cached_question"] is None


def test_semantic_paraphrase_is_served_from_cache_and_names_the_matched_question(cached):
    agent, *_ = cached
    ask(agent, "What is an ETF?")
    hit_ = ask(agent, "Could you explain what an ETF is?")
    assert hit_.cache_hit == "semantic" and agent.llm.calls == 1
    assert hit_.data["cached_question"] == "What is an ETF?" and hit_.data["similarity"] == pytest.approx(0.96, abs=1e-3)
    assert 'similar question ("What is an ETF?")' in hit_.notice


def test_cache_is_shared_across_users(cached):
    agent, *_ = cached
    ask(agent, "What is an ETF?", user_id="learner-1")
    assert ask(agent, "What is an ETF?", user_id="learner-2").cache_hit == "exact" and agent.llm.calls == 1


def test_context_dependent_questions_are_never_cached(cached):
    agent, _, cache = cached
    ask(agent, "And for bonds?")
    ask(agent, "And for bonds?")
    assert agent.llm.calls == 2 and cache.stats()["entries"] == 0


def test_questions_flagged_uncacheable_never_touch_the_cache(cached):
    agent, _, cache = cached
    ask(agent, "What is an ETF?")
    personal = ask(agent, "What is an ETF?", cacheable=False)
    assert personal.cache_hit is None and agent.llm.calls == 2
    ask(agent, "What is a bond and why do people hold them?", cacheable=False)
    assert cache.stats()["entries"] == 1


def test_refusals_are_not_cached(tmp_path, clock):
    db = Database(tmp_path / "r.db")
    db.init_schema()
    cache = QACache(db, clock=lambda: clock[0])
    agent = make_agent(FixedStore([hit("X", 0.05)]), cache=cache, embed=VecEmbedder({}))
    result = ask(agent, "What is the best pizza recipe in Chicago?")
    assert result.content == NO_ANSWER and result.cache_hit is None and cache.stats()["entries"] == 0


def test_category_is_part_of_the_cache_key(cached):
    agent, *_ = cached
    ask(agent, "What is an ETF?", category="funds")
    assert ask(agent, "What is an ETF?", category="tax").cache_hit is None
    assert ask(agent, "What is an ETF?", category=" FUNDS ").cache_hit == "exact"


def test_knowledge_base_change_invalidates_cached_answers(cached):
    agent, _, cache = cached
    ask(agent, "What is an ETF?")
    cache.bump_version()
    assert ask(agent, "What is an ETF?").cache_hit is None and agent.llm.calls == 2


def test_embedding_outage_still_answers_and_still_serves_exact_hits(tmp_path, clock):
    db = Database(tmp_path / "o.db")
    db.init_schema()
    cache = QACache(db, clock=lambda: clock[0])

    def down(_):
        raise RuntimeError("embeddings unavailable")

    agent = make_agent(FixedStore([hit("ETF", 0.7)]), cache=cache, embed=down)
    assert ask(agent, "What is an ETF?").content.startswith("Generated") and cache.stats()["entries"] == 1
    assert ask(agent, "what is an etf").cache_hit == "exact" and agent.llm.calls == 1


def test_agent_without_cache_behaves_normally():
    agent = make_agent(FixedStore([hit("ETF", 0.7)]))
    ask(agent, "What is an ETF?")
    assert ask(agent, "What is an ETF?").cache_hit is None and agent.llm.calls == 2
