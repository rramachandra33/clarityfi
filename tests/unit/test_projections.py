from datetime import date

import pytest

from src.core.models import Goal, RiskTolerance as R
from src.core.projections import (
    SPREAD_PCT,
    future_value,
    inflation_adjusted,
    months_between,
    project_goal,
    required_monthly_contribution,
    yearly_path,
    years_to_reach,
)

TODAY = date(2026, 1, 1)


def simulate(principal, monthly, annual, months):
    """Independent month-by-month reference implementation."""
    r = (1 + annual) ** (1 / 12) - 1
    value = principal
    for _ in range(months):
        value = value * (1 + r) + monthly
    return value


def test_lump_sum_growth_matches_closed_form():
    assert future_value(10_000, 0, 0.05, 120) == pytest.approx(10_000 * 1.05**10, rel=1e-9)  # 16,288.95


@pytest.mark.parametrize("principal,monthly,annual,months", [
    (0, 500, 0.07, 240), (25_000, 300, 0.03, 60), (1_000, 0, 0.10, 18), (0, 0, 0.05, 12), (5_000, 100, -0.02, 36),
])
def test_future_value_matches_simulation(principal, monthly, annual, months):
    assert future_value(principal, monthly, annual, months) == pytest.approx(simulate(principal, monthly, annual, months), rel=1e-9, abs=1e-9)


def test_zero_return_is_simple_addition():
    assert future_value(1_000, 100, 0.0, 24) == pytest.approx(3_400)


def test_zero_months_returns_principal():
    assert future_value(1_234.5, 999, 0.07, 0) == 1_234.5


def test_more_return_or_time_or_contribution_means_more_money():
    base = future_value(1_000, 100, 0.05, 120)
    assert future_value(1_000, 100, 0.07, 120) > base
    assert future_value(1_000, 100, 0.05, 180) > base
    assert future_value(1_000, 150, 0.05, 120) > base


@pytest.mark.parametrize("args", [(-1, 0, 0.05, 12), (0, -1, 0.05, 12), (0, 0, 0.05, -1)])
def test_negative_inputs_rejected(args):
    with pytest.raises(ValueError):
        future_value(*args)


def test_return_of_minus_100_percent_or_worse_rejected():
    with pytest.raises(ValueError):
        future_value(100, 0, -1.0, 12)
    with pytest.raises(ValueError):
        required_monthly_contribution(1000, 0, -1.5, 12)


@pytest.mark.parametrize("target,principal,annual,months", [
    (100_000, 5_000, 0.05, 120), (50_000, 0, 0.07, 60), (20_000, 19_000, 0.03, 12), (10_000, 0, 0.0, 50),
])
def test_required_monthly_actually_reaches_target(target, principal, annual, months):
    pmt = required_monthly_contribution(target, principal, annual, months)
    assert future_value(principal, pmt, annual, months) == pytest.approx(target, rel=1e-9)


def test_required_monthly_is_zero_when_principal_alone_suffices():
    assert required_monthly_contribution(10_000, 20_000, 0.05, 60) == 0.0


def test_required_monthly_with_no_time_left_is_the_shortfall():
    assert required_monthly_contribution(10_000, 4_000, 0.05, 0) == 6_000
    assert required_monthly_contribution(10_000, 12_000, 0.05, 0) == 0


def test_years_to_reach():
    assert years_to_reach(10_000, 10_000, 0, 0.05) == 0.0
    assert years_to_reach(12_000, 0, 1_000, 0.0) == 1.0
    assert years_to_reach(1_000_000, 0, 0, 0.0) is None
    assert years_to_reach(1_000, 0, 0, 0.05) is None
    faster = years_to_reach(100_000, 0, 500, 0.08)
    slower = years_to_reach(100_000, 0, 500, 0.02)
    assert faster < slower


def test_months_between():
    assert months_between(TODAY, TODAY) == 0
    assert months_between(date(2026, 1, 1), date(2025, 1, 1)) == 0
    assert months_between(date(2026, 1, 1), date(2027, 1, 1)) == 12
    assert months_between(date(2026, 1, 1), date(2036, 1, 1)) == 120


def test_yearly_path_shapes():
    ten = yearly_path(1_000, 100, 0.05, 120)
    assert len(ten) == 11 and ten[0] == 1_000 and ten[-1] == round(future_value(1_000, 100, 0.05, 120), 2)
    assert ten == sorted(ten)
    partial = yearly_path(1_000, 100, 0.05, 18)  # 1.5 years -> year0, year1, exact end
    assert len(partial) == 3 and partial[-1] == round(future_value(1_000, 100, 0.05, 18), 2)
    assert yearly_path(1_000, 100, 0.05, 0) == [1_000]
    assert len(yearly_path(1_000, 100, 0.05, 6)) == 2


def goal(**kw):
    base = dict(name="g", target_amount=100_000, target_date=date(2036, 1, 1), current_savings=10_000,
                monthly_contribution=500, risk_tolerance=R.moderate)
    base.update(kw)
    return Goal(**base)


def test_project_goal_scenarios_ordered_and_consistent():
    p = project_goal(goal(), today=TODAY)  # goal() defaults to assumed_return_pct=0
    assert p.years == 10.0
    assert [s.label for s in p.scenarios] == ["Below your assumption", "Your assumption", "Above your assumption"]
    values = [s.projected_value for s in p.scenarios]
    assert values == sorted(values)
    assert [round(s.annual_return * 100, 1) for s in p.scenarios] == [-2.0, 0.0, 2.0]  # 0% assumed +/- the 2pt spread
    for s in p.scenarios:
        assert s.projected_value == pytest.approx(future_value(10_000, 500, s.annual_return, 120), abs=0.01)
        assert s.real_projected_value < s.projected_value  # inflation erodes purchasing power
        assert s.on_track == (s.projected_value >= 100_000)
        assert s.shortfall == pytest.approx(max(0, 100_000 - s.projected_value), abs=0.01)
        assert s.required_monthly >= 0
    assert set(p.yearly_path) == {s.label for s in p.scenarios}


def test_on_track_goal_needs_no_extra_and_off_track_needs_more():
    easy = project_goal(goal(target_amount=20_000), today=TODAY)
    # required_monthly is the TOTAL monthly amount needed; on track means the current $500 covers it
    assert all(s.on_track and s.shortfall == 0 and s.required_monthly < 500 for s in easy.scenarios)
    hard = project_goal(goal(target_amount=1_000_000), today=TODAY)
    assert not any(s.on_track for s in hard.scenarios)
    assert all(s.required_monthly > 500 for s in hard.scenarios)


def test_risk_tolerance_does_not_affect_the_scenarios():
    values = {r: project_goal(goal(risk_tolerance=r), today=TODAY).scenarios[1].projected_value for r in R}
    assert values[R.conservative] == values[R.moderate] == values[R.aggressive]


def test_goal_in_the_past_or_today_has_no_growth():
    p = project_goal(goal(target_date=TODAY, target_amount=50_000), today=TODAY)
    assert p.years == 0 and all(s.projected_value == 10_000 and s.real_projected_value == 10_000 for s in p.scenarios)
    assert all(not s.on_track and s.required_monthly == 40_000 for s in p.scenarios)
    past = project_goal(goal(target_date=date(2020, 1, 1)), today=TODAY)
    assert past.years == 0 and past.yearly_path["Your assumption"] == [10_000]


def test_zero_savings_and_zero_contribution_goal():
    p = project_goal(goal(current_savings=0, monthly_contribution=0), today=TODAY)
    assert all(s.projected_value == 0 and s.years_to_goal is None for s in p.scenarios)


def test_goal_validation():
    with pytest.raises(Exception):
        goal(target_amount=0)
    with pytest.raises(Exception):
        goal(monthly_contribution=-5)


def test_inflation_adjusted():
    assert inflation_adjusted(1_000, 0) == 1_000
    assert inflation_adjusted(1_025, 1, 0.025) == pytest.approx(1_000)
    with pytest.raises(ValueError):
        inflation_adjusted(1, -1)


# ------------------------------------------------------------------------ user's own assumed return
def test_default_assumed_return_is_zero_percent_no_growth():
    p = project_goal(goal(current_savings=10_000, monthly_contribution=500), today=TODAY)  # 120 months, default assumed_return_pct=0
    mid = next(s for s in p.scenarios if s.label == "Your assumption")
    assert mid.annual_return == 0.0
    assert mid.projected_value == pytest.approx(10_000 + 500 * 120, abs=0.01)  # flat: principal + contributions, no growth


def test_custom_scenarios_are_centered_on_the_stated_assumed_return_with_a_fixed_spread():
    p = project_goal(goal(assumed_return_pct=6.0), today=TODAY)
    low, mid, high = p.scenarios
    assert (round(low.annual_return * 100, 1), round(mid.annual_return * 100, 1), round(high.annual_return * 100, 1)) == \
        (6.0 - SPREAD_PCT, 6.0, 6.0 + SPREAD_PCT)
    for s in (low, mid, high):
        assert s.projected_value == pytest.approx(future_value(10_000, 500, s.annual_return, 120), abs=0.01)
    assert low.projected_value < mid.projected_value < high.projected_value


def test_negative_assumed_return_still_projects_without_error():
    p = project_goal(goal(assumed_return_pct=-10.0), today=TODAY)
    low = next(s for s in p.scenarios if s.label == "Below your assumption")
    assert round(low.annual_return * 100, 1) == -12.0 and low.projected_value < 10_000 + 500 * 120  # loses to inflation-free flat growth


def test_goal_default_assumed_return_is_zero():
    assert Goal(target_amount=1, target_date=TODAY).assumed_return_pct == 0.0
