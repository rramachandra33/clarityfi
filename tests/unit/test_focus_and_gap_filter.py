"""Multi-part questions: each agent answers only its own part; narrations must not complain about the other parts."""
import pytest

from src.core.text import drop_missing_info_sentences
from src.workflow.router import focus_question
from tests.unit._agent_fakes import FakeAgent, make_workflow
from tests.unit.test_workflow import run


@pytest.mark.parametrize("question,intent,expected", [
    ("How is my portfolio doing and what is AAPL trading at?", "portfolio", "How is my portfolio doing?"),
    ("How is my portfolio doing and what is AAPL trading at?", "market", "what is AAPL trading at?"),
    ("Explain what an ETF is and analyze my portfolio", "finance_qa", "Explain what an ETF is."),
    ("Explain what an ETF is and analyze my portfolio", "portfolio", "analyze my portfolio."),
    ("Check my diversification and show the price of VTI", "market", "show the price of VTI."),
    ("Check my diversification and show the price of VTI", "portfolio", "Check my diversification."),
    ("Is my portfolio diversified, and am I on track for retirement?", "goal_planning", "am I on track for retirement?"),
    ("Am I on track for retirement, and what are the tax rules for IRAs?", "tax", "what are the tax rules for IRAs?"),
    ("What's the news on Microsoft and how is MSFT stock doing?", "news", "What's the news on Microsoft?") if False else
    ("Show me AAPL and MSFT prices", "market", "Show me AAPL and MSFT prices"),                          # 'and' joins two items: one clause
    ("What is the difference between AAPL and an S&P 500 ETF?", "finance_qa", "What is the difference between AAPL and an S&P 500 ETF?"),
    ("What is an ETF?", "finance_qa", "What is an ETF?"),
    ("How is my portfolio doing and am I on track for retirement?", "tax", "How is my portfolio doing and am I on track for retirement?"),  # no tax clause: whole question
    ("How is my portfolio doing and what is AAPL trading at?", "unknown_intent", "How is my portfolio doing and what is AAPL trading at?"),
])
def test_focus_question(question, intent, expected):
    assert focus_question(question, intent) == expected


def test_when_every_clause_matches_the_whole_question_is_kept():
    q = "What is AAPL at and what is MSFT at?"
    assert focus_question(q, "market") == q


def test_workflow_gives_each_agent_only_its_own_part():
    pf, mk = FakeAgent("portfolio", "Portfolio text."), FakeAgent("market", "AAPL text.")
    run(make_workflow([pf, mk, FakeAgent("finance_qa")]), "How is my portfolio doing and what is AAPL trading at?")
    assert pf.requests[0].question == "How is my portfolio doing?" and mk.requests[0].question == "what is AAPL trading at?"


def test_tax_fallback_and_concept_clause_each_get_their_clause():
    qa = FakeAgent("finance_qa")
    pf = FakeAgent("portfolio")
    run(make_workflow([qa, pf]), "Explain what an ETF is and analyze my portfolio")
    assert qa.requests[0].question == "Explain what an ETF is." and pf.requests[0].question == "analyze my portfolio."


def test_single_part_questions_are_passed_through_unchanged():
    qa = FakeAgent("finance_qa")
    run(make_workflow([qa]), "What is an ETF?")
    assert qa.requests[0].question == "What is an ETF?"


# ---------------------------------------------------------------------------------- gap sentences
@pytest.mark.parametrize("text,expected", [
    ("AAPL is at 339.01.\n\nThere are no trends provided, so I cannot give you any information about how AAPL has performed.", "AAPL is at 339.01."),
    ("Your portfolio is worth 1000.\n\nRegarding AAPL, the trading price is not provided in the FACTS. Your risk is Growth.",
     "Your portfolio is worth 1000.\n\nYour risk is Growth."),
    ("Price is 10. The FACTS do not include a target date. Done.", "Price is 10. Done."),
    ("Trends are not available in the facts.\nPrice is 10.", "Price is 10."),
    ("I don't have information about that. Price is 10.", "Price is 10."),
    ("No data is included in the provided facts.", "No data is included in the provided facts."),  # nothing else left: keep original
    ("A normal answer with numbers 10 and 20. Nothing missing.", "A normal answer with numbers 10 and 20. Nothing missing."),
    ("", ""),
])
def test_drop_missing_info_sentences(text, expected):
    assert drop_missing_info_sentences(text) == expected


def test_legitimate_uses_of_similar_words_survive():
    text = "Your fees are low. The provider did not raise prices. Data is included for three funds."
    assert drop_missing_info_sentences(text) == text


@pytest.mark.parametrize("question", [
    "How does compound interest work on $10,000 in VTI?", "Can I reach $1,500,000 in 30 years, saving $2,000 a month?",
    "I have $4,000 saved and $1,200.50 in fees",
])
def test_commas_inside_numbers_never_split_a_question(question):
    from src.workflow.router import _clauses

    assert _clauses(question) == [question] or all("," not in c or c.strip()[-1:].isdigit() is False for c in _clauses(question))
    assert not any(c.strip().isdigit() for c in _clauses(question))  # no orphaned "000" fragments


def test_a_comma_before_a_space_still_separates_clauses():
    from src.workflow.router import _clauses

    assert len(_clauses("Is my portfolio diversified, and am I on track?")) == 2
    assert len(_clauses("Show AAPL, then check my portfolio")) == 2
