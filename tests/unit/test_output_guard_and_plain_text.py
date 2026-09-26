"""Regressions found in the live workflow run: the output filter mangled prices, markdown leaked into a plain-text UI."""
import pytest

from src.core.pii import PiiGuard, redact_output, redact_text
from src.core.text import to_plain_text
from src.rag.categories import CATEGORY_GROUPS, resolve_categories
from tests.unit._agent_fakes import FakeAgent, make_workflow
from tests.unit.test_workflow import run


# ---------------------------------------------------------------------------------- output redaction
@pytest.mark.parametrize("text", [
    "TSLA: $364.27 (+1.02% vs the previous close) (source: Alpha Vantage, as of 2026-09-18).",
    "AAPL: $336.13 (-0.26% vs the previous close).", "VTI 70.1%, VXUS 18.5%, BND 9.2%.",
    "Warren Buffett and John Bogle favored index funds.", "Diversification score: 68 out of 100.",
])
def test_output_filter_leaves_prices_percentages_and_names_alone(text):
    result = redact_output(text)
    assert result.text == text and not result.redacted


def test_the_input_filter_still_detects_names_but_output_does_not():
    text = "I'm John Smith and the price moved +1.02% vs the previous close"
    assert "John Smith" not in redact_text(text).text
    assert redact_output("Contact John Smith about it").text == "Contact John Smith about it"  # names: not redacted on output


@pytest.mark.parametrize("text,gone", [
    ("Email advisor@example.com for details", "advisor@example.com"), ("Call 415-555-0199 today", "555-0199"),
    ("SSN 536-90-4399 on file", "536-90-4399"), ("Card 4111 1111 1111 1111", "4111 1111 1111 1111"),
    ("Server 192.168.1.10 replied", "192.168.1.10"), ("my key is sk-proj-Abcdefghijklmnopqrstuvwxyz0123456789", "sk-proj"),
])
def test_output_filter_still_removes_high_precision_identifiers(text, gone):
    assert gone not in redact_output(text).text


def test_entities_override_is_honoured_and_defaults_are_unchanged():
    guard = PiiGuard()
    assert "john@example.com" in guard.redact("john@example.com", entities=["PERSON"]).text  # email not in the override
    assert "john@example.com" not in guard.redact("john@example.com").text


def test_workflow_does_not_mangle_a_market_line_through_the_output_guard():
    line = "TSLA: $364.27 (+1.02% vs the previous close) (source: Alpha Vantage)."
    r = run(make_workflow([FakeAgent("market", line)]), "What is TSLA trading at?")
    assert line in r.content and "<PERSON>" not in r.content


# ------------------------------------------------------------------------------------- plain text
@pytest.mark.parametrize("markdown,plain", [
    ("This is **bold** text", "This is bold text"), ("**Diversification Score**: 68", "Diversification Score: 68"),
    ("# Heading\nBody", "Heading\nBody"), ("### Sub heading", "Sub heading"),
    ("* item one\n* item two", "- item one\n- item two"), ("  * nested", "- nested"), ("• dot bullet", "- dot bullet"),
    ("Use `VTI` here", "Use VTI here"), ("__strong__ word", "strong word"),
    ("```\ncode\n```", "code"), ("Line\n\n\n\n\nEnd", "Line\n\nEnd"),
    ("1. **Define Your Goals**: think", "1. Define Your Goals: think"),
])
def test_markdown_is_removed(markdown, plain):
    assert to_plain_text(markdown) == plain


@pytest.mark.parametrize("text", [
    "5 * 3 = 15 and 2 * 4 = 8", "Plain sentence with no markup.", "- already a dash bullet", "snake_case_name stays", "A_B and C_D",
    "Price is $10 * 2 shares", "1. numbered list\n2. keeps numbers",
])
def test_ordinary_text_is_not_touched(text):
    assert to_plain_text(text) == text


def test_qa_and_narration_outputs_are_plain_text():
    import asyncio

    from src.agents.narration import narrate
    from tests.unit._agent_fakes import FakeLLM, FixedStore, hit, make_agent, ask

    agent = make_agent(FixedStore([hit("ETF", 0.7)]), FakeLLM("**An ETF** is a basket.\n\n* one\n* two [1]"))
    content = ask(agent, "What is an ETF?").content
    assert "**" not in content and "* " not in content and "- one" in content

    text, mode = asyncio.run(narrate("q", {"total": 1000}, lambda: "fallback", llm_factory=lambda: FakeLLM("Your total is **$1,000**."),
                                     callbacks_factory=lambda: []))
    assert mode == "llm" and text.startswith("Your total is $1,000.")  # markdown stripped BEFORE the grounding check


# ------------------------------------------------------------------------------------ tax category
def test_tax_group_includes_the_retirement_account_articles():
    assert "retirement" in CATEGORY_GROUPS["tax"] and set(CATEGORY_GROUPS["tax"]) >= {"taxes_and_investing", "tax_efficiency", "tax_education"}
    assert resolve_categories("tax") == list(CATEGORY_GROUPS["tax"])
