import pytest

from src.agents.tax import PERSONAL_TAX_REDIRECT, TAX_NOTICE, TAX_SYSTEM_PROMPT, TaxEducationAgent, is_personal_tax_question
from src.agents.finance_qa import NO_ANSWER
from src.core.models import AgentName
from src.data.db import Database
from src.rag.qa_cache import QACache
from tests.unit._agent_fakes import FakeLLM, FixedStore, ask, hit


def tax_agent(store=None, llm=None, cache=None, embed=None, threshold=0.25):
    store = store or FixedStore([hit("Roth IRA", 0.7, source_name="Investopedia", source_url="https://ex.com/roth", category="retirement")])
    llm = llm or FakeLLM("A Roth IRA is funded with after-tax money, so qualified withdrawals can be tax free. [1]")
    a = TaxEducationAgent(store, cache=cache, embed=embed, score_threshold=threshold, llm_factory=lambda: llm, callbacks_factory=lambda: [])
    a.llm, a.store = llm, store
    return a


def test_reads_the_tax_slice_of_the_knowledge_base_with_a_strict_prompt():
    a = tax_agent()
    r = ask(a, "What is a Roth IRA?")
    assert r.ok and r.agent is AgentName.tax and a.store.calls[0]["category"] == "tax"
    assert a.llm.messages[0][0].content == TAX_SYSTEM_PROMPT and "Never calculate a person's tax bill" in TAX_SYSTEM_PROMPT
    assert "Do not state a specific dollar limit" in TAX_SYSTEM_PROMPT
    assert r.citations[0].source == "Investopedia" and r.citations[0].url == "https://ex.com/roth"


def test_an_explicit_category_overrides_the_default():
    a = tax_agent()
    ask(a, "What is a Roth IRA?", category="retirement")
    assert a.store.calls[0]["category"] == "retirement"


def test_every_answer_carries_the_rules_change_caveat_exactly_once():
    r = ask(tax_agent(), "What is a Roth IRA?")
    assert r.content.count(TAX_NOTICE) == 1 and "not tax advice" in r.content and r.data["personal_tax_redirect"] is False
    assert PERSONAL_TAX_REDIRECT not in r.content


@pytest.mark.parametrize("question", [
    "How much tax will I owe on selling my VTI shares?", "Should I convert my traditional IRA to a Roth?", "Can I deduct my home office?",
    "What's my tax bracket?", "Do I qualify for the saver's credit?", "Am I eligible to contribute to a Roth IRA?",
    "Will I get a refund this year?", "Is my side income taxable?", "Should I itemize on my return?", "calculate my capital gains tax",
])
def test_personal_tax_questions_are_redirected_but_still_educated(question):
    assert is_personal_tax_question(question) is True
    r = ask(tax_agent(), question)
    assert PERSONAL_TAX_REDIRECT in r.content and r.content.count(TAX_NOTICE) == 1 and r.data["personal_tax_redirect"] is True
    assert r.content.index("A Roth IRA is funded") < r.content.index(PERSONAL_TAX_REDIRECT)  # the explanation comes first


@pytest.mark.parametrize("question", [
    "What is a Roth IRA?", "How are capital gains taxed?", "What is the difference between a 401k and an IRA?", "How does an HSA work?",
    "Why do people say tax-loss harvesting helps?", "What are qualified dividends?",
])
def test_general_tax_education_is_not_treated_as_personal(question):
    assert is_personal_tax_question(question) is False


def test_refusals_and_errors_are_not_decorated():
    refusal = ask(tax_agent(FixedStore([hit("Unrelated", 0.05)])), "What is the best pizza in Chicago?")
    assert refusal.content == NO_ANSWER and TAX_NOTICE not in refusal.content and "personal_tax_redirect" not in refusal.data
    empty = ask(tax_agent(FixedStore([], count=0)), "What is a Roth IRA?")
    assert not empty.ok and empty.error_code == "knowledge_base_empty"


def test_cached_answers_get_the_same_notes_once_and_are_cached_under_the_tax_scope(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    cache = QACache(db)
    a = tax_agent(cache=cache, embed=lambda t: [1.0, 0, 0, 0])
    first = ask(a, "What is a Roth IRA?")
    second = ask(a, "what is a roth ira")
    assert second.cache_hit == "exact" and a.llm.calls == 1 and second.notice and "same question" in second.notice
    assert second.content.count(TAX_NOTICE) == 1 and first.content == second.content
    assert cache.lookup("What is a Roth IRA?", scope="tax") is not None and cache.lookup("What is a Roth IRA?", scope="") is None
    with cache.db.connect() as conn:  # the stored answer itself does not contain the decorations
        assert TAX_NOTICE not in conn.execute("SELECT answer FROM qa_cache").fetchone()[0]


def test_inherits_the_plain_text_and_disclaimer_guarantees():
    r = ask(tax_agent(llm=FakeLLM("**Roth IRA** basics:\n* after-tax money [1]")), "What is a Roth IRA?")
    assert "**" not in r.content and "- after-tax money" in r.content and "educational" in r.content.lower()
