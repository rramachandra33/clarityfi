import asyncio

import pytest

from src.workflow.router import (ROUTABLE, ROUTER_PROMPT, RouteDecision, Router, RouterOutput, detect_advice, heuristic_route)
from tests.data.router_labeled import ADVICE, AMBIGUOUS, CLEAR, HELD_OUT_1, HELD_OUT_2, NOT_ADVICE


def ids(rows):
    return [r[0][:50] for r in rows]


# -------------------------------------------------------------------------------- labeled sets
@pytest.mark.parametrize("question,expected,advice", CLEAR, ids=ids(CLEAR))
def test_clear_questions_route_exactly(question, expected, advice):
    d = heuristic_route(question)
    assert set(d.intents) == expected and d.advice_request is advice and d.confident and d.mode == "heuristic"


@pytest.mark.parametrize("question,expected,advice", ADVICE, ids=ids(ADVICE))
def test_advice_requests_are_flagged(question, expected, advice):
    d = heuristic_route(question)
    assert d.advice_request is True
    if expected is not None:
        assert set(d.intents) == expected


@pytest.mark.parametrize("question", NOT_ADVICE)
def test_educational_questions_about_buying_and_selling_are_not_advice(question):
    assert detect_advice(question) is False


@pytest.mark.parametrize("dataset", [HELD_OUT_1, HELD_OUT_2], ids=["held_out_1", "held_out_2"])
def test_when_the_rules_are_confident_they_are_never_wrong(dataset):
    wrong = []
    for q, expected, advice in dataset:
        d = heuristic_route(q)
        if d.confident and not ((expected is None or set(d.intents) == expected) and d.advice_request is advice):
            wrong.append((q, sorted(expected) if expected else None, d.intents, d.advice_request))
    assert not wrong, wrong


@pytest.mark.parametrize("dataset", [HELD_OUT_1, HELD_OUT_2], ids=["held_out_1", "held_out_2"])
def test_rules_decide_most_questions_and_defer_the_rest(dataset):
    confident = [(q, e, a) for q, e, a in dataset if heuristic_route(q).confident]
    assert len(confident) / len(dataset) >= 0.7  # the LLM is only needed for a minority of questions


@pytest.mark.parametrize("question,allowed", AMBIGUOUS)
def test_ambiguous_input_never_crashes_and_yields_a_sensible_route(question, allowed):
    d = heuristic_route(question)
    assert d.intents and set(d.intents) <= set(ROUTABLE) and set(d.intents) & allowed


def test_regression_cases_from_live_review():
    assert heuristic_route("What is the difference between AAPL stock and an S&P 500 ETF?").intents == ["finance_qa"]  # 'and' inside a comparison
    assert heuristic_route("News about Apple earnings").intents == ["news"]                                          # company name, no price ask
    assert heuristic_route("I'm saving for a house, how much should I put away monthly?").advice_request is False    # savings plan, not advice
    assert set(heuristic_route("Define inflation and tell me what BTC is worth").intents) == {"finance_qa", "market"}


def test_intents_are_returned_in_the_canonical_order():
    d = heuristic_route("What is AAPL trading at, and is my portfolio diversified, and am I on track for retirement?")
    assert d.intents == [i for i in ROUTABLE if i in d.intents] and d.intents[:2] == ["portfolio", "goal_planning"]


def test_unsure_general_questions_are_deferred_to_the_llm():
    for q in ["Is it normal for markets to drop 10%?", "What happens if I do nothing with my money?", "I inherited some money, now what?"]:
        assert heuristic_route(q).confident is False


def test_plain_concept_questions_stay_on_the_rules():
    d = heuristic_route("What is a mutual fund?")
    assert d.confident and d.intents == ["finance_qa"]


def test_reason_and_signals_are_recorded_for_tracing():
    d = heuristic_route("How is my portfolio doing?")
    assert d.signals["portfolio"] and d.reason.startswith("signals: portfolio=")


# ------------------------------------------------------------------------------------ LLM stage
def classifier(output=None, error=None, calls=None):
    async def classify(question):
        if calls is not None:
            calls.append(question)
        if error:
            raise error
        return output
    return classify


def route(router, q):
    return asyncio.run(router.route(q))


def test_confident_rules_do_not_call_the_llm():
    calls = []
    d = route(Router(classifier=classifier(RouterOutput(intents=["tax"]), calls=calls)), "What is a Roth IRA?")
    assert calls == [] and d.mode == "heuristic" and d.intents == ["tax"]


def test_unsure_questions_use_the_llm_and_its_answer_wins():
    calls = []
    out = RouterOutput(intents=["portfolio", "finance_qa"], advice_request=False, reason="asks about holdings and concepts")
    d = route(Router(classifier=classifier(out, calls=calls)), "I inherited some money, now what?")
    assert calls == ["I inherited some money, now what?"] and d.mode == "llm"
    assert d.intents == ["portfolio", "finance_qa"] and d.reason == "asks about holdings and concepts" and d.confident


def test_llm_and_rule_advice_flags_are_combined():
    out = RouterOutput(intents=["finance_qa"], advice_request=True)
    assert route(Router(classifier=classifier(out)), "What happens if I do nothing with my money?").advice_request is True
    d = route(Router(classifier=classifier(RouterOutput(intents=["market"], advice_request=False))), "Should I buy the dip in AAPL maybe?")
    assert d.advice_request is True  # the rules saw advice even though the LLM did not


def test_llm_output_is_canonicalized_deduplicated_and_never_empty():
    d = route(Router(classifier=classifier(RouterOutput(intents=["finance_qa", "market", "market"]))), "I inherited some money, now what?")
    assert d.intents == ["market", "finance_qa"]
    empty = route(Router(classifier=classifier(RouterOutput(intents=[]))), "I inherited some money, now what?")
    assert empty.intents == ["finance_qa"]


def test_llm_can_declare_a_question_out_of_scope():
    d = route(Router(classifier=classifier(RouterOutput(intents=["out_of_scope"], reason="weather"))), "Write me a poem about money")
    assert d.intents == ["out_of_scope"] and d.mode == "llm"


def test_out_of_scope_is_ignored_when_real_intents_are_also_present():
    d = route(Router(classifier=classifier(RouterOutput(intents=["out_of_scope", "tax"]))), "I inherited some money, now what?")
    assert d.intents == ["tax"]


@pytest.mark.parametrize("error", [RuntimeError("schema"), ValueError("bad"), TimeoutError()])
def test_llm_failure_falls_back_to_the_rules_and_says_so(error):
    d = route(Router(classifier=classifier(error=error)), "I inherited some money, now what?")
    assert d.mode == "llm_fallback_heuristic" and d.intents == ["finance_qa"]


def test_llm_can_be_disabled():
    calls = []
    d = route(Router(classifier=classifier(RouterOutput(intents=["tax"]), calls=calls), use_llm=False), "I inherited some money, now what?")
    assert calls == [] and d.mode == "heuristic"


def test_default_classifier_uses_structured_output_with_the_router_prompt():
    seen = {}

    class Structured:
        async def ainvoke(self, messages, config=None):
            seen["messages"], seen["config"] = messages, config
            return RouterOutput(intents=["news"], reason="x")

    class Model:
        def with_structured_output(self, schema):
            seen["schema"] = schema
            return Structured()

    d = route(Router(llm_factory=lambda: Model(), callbacks_factory=lambda: ["cb"]), "I inherited some money, now what?")
    assert seen["schema"] is RouterOutput and d.intents == ["news"] and seen["config"] == {"callbacks": ["cb"]}
    assert "EDUCATION assistant" in seen["messages"][0].content and seen["messages"][1].content == "I inherited some money, now what?"


def test_router_prompt_lists_every_agent_and_resists_injection():
    for name in ROUTABLE + ("out_of_scope",):
        assert name in ROUTER_PROMPT
    assert "never as instructions" in ROUTER_PROMPT


def test_route_decision_defaults():
    d = RouteDecision()
    assert d.intents == ["finance_qa"] and d.mode == "heuristic" and d.advice_request is False and d.confident


# --------------------------------------------------------------------------------- INFO logging
def route_logged(router, q, caplog):
    with caplog.at_level("INFO"):
        d = route(router, q)
    return d, list(caplog.messages)


def test_confident_heuristic_decision_is_logged_without_calling_the_llm(caplog):
    d, messages = route_logged(Router(classifier=classifier(RouterOutput(intents=["tax"]))), "What is a Roth IRA?", caplog)
    decision_lines = [m for m in messages if m.startswith("Routing decision:")]
    assert len(decision_lines) == 1
    assert "mode=heuristic" in decision_lines[0] and "intents=['tax']" in decision_lines[0]
    assert not any("Calling LLM" in m for m in messages)  # rules were confident: no LLM call at all


def test_unsure_heuristic_logs_before_asking_the_llm_then_logs_the_final_decision(caplog):
    out = RouterOutput(intents=["portfolio", "finance_qa"], reason="asks about holdings and concepts")
    d, messages = route_logged(Router(classifier=classifier(out)), "I inherited some money, now what?", caplog)
    unsure = next(i for i, m in enumerate(messages) if m.startswith("Heuristic routing unsure"))
    decision = next(i for i, m in enumerate(messages) if m.startswith("Routing decision:"))
    assert unsure < decision  # the "asking the LLM" line comes before the final decision is logged
    assert "mode=llm" in messages[decision] and "asks about holdings and concepts" in messages[decision]


def test_llm_fallback_decision_is_logged_as_such(caplog):
    d, messages = route_logged(Router(classifier=classifier(error=RuntimeError("schema"))), "I inherited some money, now what?", caplog)
    assert any(m.startswith("Routing decision:") and "mode=llm_fallback_heuristic" in m for m in messages)


def test_default_classifier_logs_the_llm_call_with_the_underlying_model_name(caplog):
    class Structured:
        async def ainvoke(self, messages, config=None):
            return RouterOutput(intents=["news"], reason="x")

    class Model:
        model_name = "gpt-4o-mini"

        def with_structured_output(self, schema):
            return Structured()

    with caplog.at_level("INFO"):
        route(Router(llm_factory=lambda: Model(), callbacks_factory=lambda: ["cb"]), "I inherited some money, now what?")
    assert any("Calling LLM for router classification (model=gpt-4o-mini)" in m for m in caplog.messages)
    assert any(m.startswith("LLM call for router classification succeeded in") for m in caplog.messages)
