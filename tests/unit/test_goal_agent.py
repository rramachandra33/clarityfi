import json
from datetime import date

import pytest

from src.agents.base import AgentRequest
from src.agents.goal_planning import (MAX_TARGET, NEEDS_GOAL, GoalParams, GoalPlanningAgent, numbers_in_text, verify_extraction)
from src.core.errors import LLMError
from src.core.models import Goal, RiskTolerance
from src.core.projections import project_goal
from src.data.db import Database
from src.data.seed import seed
from tests.unit._agent_fakes import FakeLLM

TODAY = date(2026, 9, 21)


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "goal.db")
    d.init_schema()
    seed(d)
    return d


def agent(db, llm=None, extractor=None):
    llm = llm or FakeLLM("You are on track under the expected scenario.")
    a = GoalPlanningAgent(db, llm_factory=lambda: llm, callbacks_factory=lambda: [], extractor=extractor, today=lambda: TODAY)
    a.llm = llm
    return a


def go(a, question="Am I on track?", user="learner-101", **kw):
    return a.run(AgentRequest(question=question, user_id=user, **kw))


def extractor_returning(params, calls=None):
    async def extract(question):
        if calls is not None:
            calls.append(question)
        return params
    return extract


def facts_sent(a):
    return json.loads(a.llm.messages[0][-1].content.split("FACTS:\n", 1)[1])


GOAL = {"name": "House", "target_amount": 50000, "target_date": "2031-09-21", "current_savings": 4000, "monthly_contribution": 300,
        "risk_tolerance": "moderate"}


# ------------------------------------------------------------------------------- numbers / verification
@pytest.mark.parametrize("text,expected", [
    ("I want $50,000 in 5 years", {50000, 5}), ("save $1.5 million", {1.5, 1500000}), ("a $50k house", {50, 50000}),
    ("$300 a month, 4% return", {300, 4}), ("no numbers", set()), ("2 thousand", {2, 2000}),
])
def test_numbers_in_text(text, expected):
    assert numbers_in_text(text) == expected


def test_verification_drops_values_the_user_never_stated():
    q = "I want $50,000 in 5 years and can add $300 a month"
    out = verify_extraction(GoalParams(target_amount=50000, years=5, current_savings=99999, monthly_contribution=300), q)
    assert (out.target_amount, out.years, out.monthly_contribution, out.current_savings) == (50000, 5, 300, None)


def test_verification_accepts_k_and_million_forms_and_drops_unmentioned_dates():
    assert verify_extraction(GoalParams(target_amount=50000), "a $50k goal").target_amount == 50000
    assert verify_extraction(GoalParams(target_amount=1_500_000), "reach $1.5 million").target_amount == 1_500_000
    q = "I want 20000 by 2030"
    assert verify_extraction(GoalParams(target_amount=20000, target_date=date(2030, 6, 1)), q).target_date == date(2030, 6, 1)
    assert verify_extraction(GoalParams(target_amount=20000, target_date=date(2041, 6, 1)), q).target_date is None


@pytest.mark.parametrize("kwargs", [{"target_amount": -1}, {"target_amount": MAX_TARGET * 2}, {"years": 0}, {"years": 200},
                                    {"monthly_contribution": -5}])
def test_goal_params_reject_absurd_values(kwargs):
    with pytest.raises(Exception):
        GoalParams(**kwargs)


# --------------------------------------------------------------------------------------- projections
def test_provided_goal_is_projected_by_code_and_narrated(db):
    goal = Goal.model_validate(GOAL)
    expected = project_goal(goal, today=TODAY)
    mid = expected.scenarios[1]
    a = agent(db, FakeLLM(f"Under the expected case you would reach about ${mid.projected_value:,.0f}."))
    r = go(a, context={"goal": GOAL})
    assert r.ok and r.data["narration"] == "llm" and r.data["goal_source"] == "provided"
    assert r.data["projection"]["scenarios"][1]["projected_value"] == mid.projected_value
    assert r.data["on_track_expected"] == mid.on_track and len(r.data["projection"]["yearly_path"]) == 3
    f = facts_sent(a)
    assert f["scenarios"][1]["projected_value"] == mid.projected_value and f["risk_tolerance"] == "moderate"
    assert any("not forecasts or guarantees" in x for x in f["assumptions"])


def test_extraction_from_the_question_builds_a_goal_and_states_defaults(db):
    calls = []
    q = "I want $20,000 in 4 years"
    a = agent(db, extractor=extractor_returning(GoalParams(name="car", target_amount=20000, years=4), calls))
    r = go(a, q)
    goal = r.data["goal"]
    assert calls == [q] and r.data["goal_source"] == "question"
    assert goal["target_amount"] == 20000 and goal["current_savings"] == 0 and goal["monthly_contribution"] == 0
    assert goal["target_date"] == "2030-09-21" or goal["target_date"] == "2030-09-20"  # 4 years from today
    assert goal["risk_tolerance"] == "aggressive"  # learner-101's profile
    assert any("assumed $0" in x and "already saved" in x for x in r.data["assumptions"])
    assert any("assumed $0 per month" in x for x in r.data["assumptions"])


def test_stated_values_are_used_and_no_default_assumptions_are_added(db):
    q = "I have $4,000 saved, add $300 a month, and want $50,000 in 5 years"
    a = agent(db, extractor=extractor_returning(GoalParams(target_amount=50000, years=5, current_savings=4000, monthly_contribution=300)))
    r = go(a, q)
    assert r.data["goal"]["current_savings"] == 4000 and r.data["goal"]["monthly_contribution"] == 300
    assert not any("assumed" in x for x in r.data["assumptions"])


def test_hallucinated_extraction_falls_back_to_saved_goals(db):
    a = agent(db, extractor=extractor_returning(GoalParams(target_amount=777777, years=9)))
    r = go(a, "How am I doing with 5 things?")  # digits present, but 777777 and 9 are not in the message
    assert r.data["goal_source"] == "saved" and r.data["goal"]["name"] == "Down payment"


def test_risk_tolerance_stated_in_the_question_overrides_the_profile(db):
    q = "I want $10,000 in 3 years and I'm conservative"
    a = agent(db, extractor=extractor_returning(GoalParams(target_amount=10000, years=3, risk_tolerance=RiskTolerance.conservative)))
    assert go(a, q).data["goal"]["risk_tolerance"] == "conservative"


def test_no_digits_means_no_extraction_call_and_saved_goal_is_used(db):
    calls = []
    a = agent(db, extractor=extractor_returning(GoalParams(target_amount=1, years=1), calls))
    r = go(a, "Am I on track for my down payment?")
    assert calls == [] and r.data["goal_source"] == "saved" and r.data["goal"]["name"] == "Down payment"


def test_saved_goal_is_matched_by_name_else_first(db):
    a = agent(db)
    assert go(a, "How is my retirement going?", user="learner-108").data["goal"]["name"] == "Retirement"
    assert go(a, "How is my renovation going?", user="learner-108").data["goal"]["name"] == "Home renovation"
    assert go(a, "How am I doing?", user="learner-108").data["goal"]["name"] == "Home renovation"  # first saved


def test_no_goal_anywhere_asks_for_details_without_calling_the_llm(db):
    a = agent(db)
    r = go(a, "Help me plan", user="learner-110")
    assert r.content == NEEDS_GOAL and r.data["needs_goal"] is True and a.llm.calls == 0


def test_invalid_provided_goal_asks_again(db):
    r = go(agent(db), context={"goal": {"target_amount": -5, "target_date": "not a date"}})
    assert r.content == NEEDS_GOAL and r.data["reason"] == "invalid_goal"


@pytest.mark.parametrize("exc", [RuntimeError("bad json"), LLMError("down"), ValueError("schema")])
def test_extraction_failure_never_falls_back_to_an_unrelated_saved_goal(db, exc):
    """Regression (found live): when the extractor failed, a saved 'Retirement' goal was silently narrated
    against a question about a $50,000 house."""
    async def broken(question):
        raise exc

    a = agent(db, extractor=broken)
    r = go(a, "I want $50,000 in 5 years for a house")
    assert r.ok and r.data["needs_goal"] is True and r.data["reason"] == "extraction_failed" and "goal" not in r.data
    assert "try again" in r.content and a.llm.calls == 0


def test_extraction_is_retried_once_before_giving_up(db):
    attempts = []

    async def flaky(question):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("transient")
        return GoalParams(target_amount=20000, years=4)

    r = go(agent(db, extractor=flaky), "I want $20,000 in 4 years")
    assert len(attempts) == 2 and r.data["goal_source"] == "question"


def test_partial_figures_without_a_saved_goal_ask_for_the_rest_and_say_what_was_understood(db):
    a = agent(db, extractor=extractor_returning(GoalParams(target_amount=50000)))
    r = go(a, "I want to save $50,000", user="learner-110")  # learner-110 has no saved goals
    assert r.data["reason"] == "incomplete" and r.data["understood"] == {"target_amount": 50000.0}
    assert "when you want to reach it" in r.data["missing"][0] and "I understood: target amount: 50000.0" in r.content.replace("So far ", "")
    assert a.llm.calls == 0


def test_named_saved_goal_fills_gaps_but_stated_figures_win(db):
    # learner-108 has a saved 'Retirement' goal ($1.4M, 2048-01-01, $72,000 saved, $1,500/month)
    a = agent(db, extractor=extractor_returning(GoalParams(target_date=date(2040, 6, 1), monthly_contribution=2500)))
    r = go(a, "Can I retire by 2040 if I save $2,500 a month?", user="learner-108")
    goal = r.data["goal"]
    assert r.data["goal_source"] == "question+saved" and goal["name"] == "Retirement"
    assert goal["target_date"] == "2040-06-01" and goal["monthly_contribution"] == 2500  # stated
    assert goal["target_amount"] == 1_400_000 and goal["current_savings"] == 72_000      # from the saved goal
    assert any("saved goal 'Retirement'" in x and "target amount" in x for x in r.data["assumptions"])


def test_figures_alone_are_never_merged_with_an_unnamed_saved_goal(db):
    a = agent(db, extractor=extractor_returning(GoalParams(target_amount=30000, years=6)))
    r = go(a, "I want $30,000 in 6 years", user="learner-108")  # the question names no saved goal
    assert r.data["goal_source"] == "question" and r.data["goal"]["current_savings"] == 0


def test_digits_but_no_goal_figures_uses_the_saved_goal_with_a_disclosure(db):
    a = agent(db, extractor=extractor_returning(None))
    r = go(a, "Am I on track for my down payment? I am 34.")
    assert r.data["goal_source"] == "saved" and any("didn't find goal figures" in x and "Down payment" in x for x in r.data["assumptions"])


def test_default_extractor_uses_structured_output_and_verifies(db):
    class Structured:
        def __init__(self):
            self.messages = None

        async def ainvoke(self, messages, config=None):
            self.messages = messages
            return GoalParams(target_amount=30000, years=6, current_savings=123456)  # 123456 is not in the question

    class Model:
        def __init__(self):
            self.structured = Structured()

        def with_structured_output(self, schema):
            assert schema is GoalParams
            return self.structured

    model = Model()
    a = GoalPlanningAgent(db, llm_factory=lambda: model, callbacks_factory=lambda: [], today=lambda: TODAY)
    import asyncio
    params, failed = asyncio.run(a._extract("I want $30,000 in 6 years"))
    assert failed is False
    assert params.target_amount == 30000 and params.years == 6 and params.current_savings is None
    assert "Never guess" in model.structured.messages[0].content and "$30,000" in model.structured.messages[1].content


def test_extraction_call_is_logged_at_info_with_the_model_name(db, caplog):
    class Structured:
        async def ainvoke(self, messages, config=None):
            return GoalParams(target_amount=30000, years=6)

    class Model:
        model_name = "gpt-4o-mini"

        def with_structured_output(self, schema):
            return Structured()

    a = GoalPlanningAgent(db, llm_factory=lambda: Model(), callbacks_factory=lambda: [], today=lambda: TODAY)
    import asyncio
    with caplog.at_level("INFO"):
        asyncio.run(a._extract("I want $30,000 in 6 years"))
    assert any("Calling LLM for goal parameter extraction (model=gpt-4o-mini, attempt=1)" in m for m in caplog.messages)
    assert any(m.startswith("LLM call for goal parameter extraction succeeded in") for m in caplog.messages)


def test_goal_in_the_past_is_handled_gracefully(db):
    past = dict(GOAL, target_date="2020-01-01")
    r = go(agent(db, FakeLLM(error=LLMError())), context={"goal": past})
    assert r.ok and r.data["projection"]["years"] == 0 and "0.0 years away" in r.content


# --------------------------------------------------------------------------------- narration safety
def test_llm_outage_returns_a_full_template_with_all_three_scenarios(db):
    r = go(agent(db, FakeLLM(error=LLMError("down"))), context={"goal": GOAL})
    assert r.data["narration"] == "template"
    for label in ("Below your assumption", "Your assumption", "Above your assumption"):
        assert label in r.content
    assert "Goal: $50,000 by 2031-09-21" in r.content and "not forecasts or guarantees" in r.content and "Educational only" in r.content


def test_invented_projection_figures_are_rejected(db):
    r = go(agent(db, FakeLLM("You will surely have $1,000,000 and 40% growth.")), context={"goal": GOAL})
    assert r.data["narration"] == "template" and "1,000,000" not in r.content


def test_user_id_and_personal_profile_never_reach_the_llm(db):
    a = agent(db)
    go(a, context={"goal": GOAL}, user="learner-101")
    human = a.llm.messages[0][-1].content
    assert "learner-101" not in human and "Alex Rivera" not in human


@pytest.mark.parametrize("question,expected", [
    ("Can I retire by 2040?", "Retirement"), ("how is the renovation going", "Home renovation"),
    ("Will I afford my home upgrade", "Home renovation"), ("what about my plans", None), ("an unrelated question here", None),
])
def test_saved_goal_matching_uses_word_stems(db, question, expected):
    from src.data.repositories import GoalRepository
    goals = GoalRepository(db).list("learner-108")
    named = GoalPlanningAgent._named(goals, question)
    assert (named.name if named else None) == expected
