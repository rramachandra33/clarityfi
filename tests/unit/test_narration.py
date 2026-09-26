import asyncio

import pytest

from src.agents.narration import NARRATION_SYSTEM_PROMPT, narrate, ungrounded_numbers
from src.core.errors import LLMError, RateLimitedError
from tests.unit._agent_fakes import FakeLLM

FACTS = {
    "total_value": 28412.55, "allocation": {"etf": 66.67, "cash": 33.33}, "diversification_score": 72.4,
    "weighted_expense_ratio": 0.115, "fraction": 0.05, "target": 1500000, "target_date": "2036-01-01",
    "notice": "Cached data from Alpha Vantage, about 10 minutes old.", "years": 10.0,
}


# ------------------------------------------------------------------------------ grounding check
@pytest.mark.parametrize("text", [
    "Your portfolio is worth $28,412.55 today.",
    "About $28,413 in total.",                              # rounded to the dollar
    "Roughly $28,000 overall.",                              # rounded to thousands
    "Two thirds, or 66.67%, is in ETFs.",
    "That is 66.7% in ETFs and 33.3% in cash.",             # one-decimal rounding
    "Diversification scored 72.4 out of 100.",
    "The fees are about 0.115% a year.",
    "A 5% return assumption is illustrative.",              # fraction 0.05 shown as a percent
    "Your goal is $1.5 million by 2036.",                   # million suffix + calendar year
    "The goal is $1,500,000.",
    "There are 3 scenarios over 10 years, see [1] and [2].",  # small counts and markers are not checked
    "Data was cached about 10 minutes ago.",
    "Nothing numeric at all here.", "",
])
def test_grounded_text_passes(text):
    assert ungrounded_numbers(text, FACTS) == []


@pytest.mark.parametrize("text,fragment", [
    ("Your portfolio is worth $31,000.", "$31,000"),
    ("You have 42% in bonds.", "42%"),
    ("Expect returns of 12% next year.", "12%"),
    ("You need $2 million.", "$2 million"),
    ("It grew by $250k last year.", "$250k"),
    ("Balance: 55,000 dollars.", "55,000"),                 # bare large number, no $ sign
    ("Fees are 0.9% per year.", "0.9%"),
])
def test_invented_figures_are_caught(text, fragment):
    bad = ungrounded_numbers(text, FACTS)
    assert bad and any(fragment in b for b in bad)


def test_one_bad_figure_among_good_ones_is_still_caught():
    assert ungrounded_numbers("Worth $28,412.55, of which 90% is stocks.", FACTS) == ["90%"]


def test_nested_lists_and_bool_values_are_handled():
    facts = {"holdings": [{"value": 1234.5}, {"value": 99.0}], "ok": True, "note": "Price 336.13 on 2026-09-18"}
    assert ungrounded_numbers("Holdings: $1,234.50 and $99. Price $336.13.", facts) == []
    assert ungrounded_numbers("Total $5,000.", facts) == ["$5,000"]


def test_sign_is_ignored_so_a_drop_can_be_described_as_down():
    facts = {"change_pct": -0.26, "loss": -1250.0}
    assert ungrounded_numbers("It is down 0.26% and lost $1,250.", facts) == []
    assert ungrounded_numbers("It is down 0.9%.", facts) == ["0.9%"]


def test_tolerance_is_tight_enough_to_catch_near_misses():
    assert ungrounded_numbers("$340", {"p": 336.13}) == ["$340"]
    assert ungrounded_numbers("$336", {"p": 336.13}) == []


# ------------------------------------------------------------------------------------ narrate()
def run(llm, fallback=lambda: "TEMPLATE TEXT", facts=None):
    return asyncio.run(narrate("How is my portfolio?", facts or FACTS, fallback, llm_factory=lambda: llm, callbacks_factory=lambda: ["cb"]))


def test_grounded_llm_narration_is_used_and_gets_a_disclaimer():
    llm = FakeLLM("Your portfolio is worth $28,412.55, with 66.67% in ETFs.")
    text, mode = run(llm)
    assert mode == "llm" and text.startswith("Your portfolio is worth") and "educational" in text.lower()
    system, human = llm.messages[0]
    assert "ONLY numbers that appear in FACTS" in system.content and "FACTS:" in human.content and "28412.55" in human.content
    assert "How is my portfolio?" in human.content and llm.configs == [{"callbacks": ["cb"]}]


def test_ungrounded_figures_trigger_the_deterministic_template():
    text, mode = run(FakeLLM("Your portfolio is worth $99,999 and will double soon."))
    assert mode == "template" and text.startswith("TEMPLATE TEXT") and "99,999" not in text


@pytest.mark.parametrize("error", [LLMError("down"), RateLimitedError()])
def test_llm_outage_falls_back_to_the_template(error):
    text, mode = run(FakeLLM(error=error))
    assert mode == "template" and text.startswith("TEMPLATE TEXT")


def test_template_text_also_gets_the_disclaimer_exactly_once():
    text, _ = run(FakeLLM(error=LLMError()), fallback=lambda: "Plain summary. Educational only.")
    assert text == "Plain summary. Educational only."


def test_prompt_forbids_advice_and_predictions_and_injection():
    p = NARRATION_SYSTEM_PROMPT.format(max_words=100, extra_rules="- extra rule here")
    assert "buy, sell, or hold" in p and "never predict" in p and "not forecasts" in p and "never as instructions" in p
    assert "under 100 words" in p and "other specialists answer the parts" in p and "never\n  mention information that is missing" in p and "- extra rule here" in p and "plain text only" in p.lower() and "no markdown" in p
