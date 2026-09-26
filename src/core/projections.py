"""Goal-planning math: compound growth with monthly contributions, a below/your-assumption/above trio centered on
the user's own assumed return.

Conventions
- Annual returns are *effective* annual rates; monthly growth is (1+r)^(1/12)-1.
- Contributions are made at the END of each month.
- Scenario return assumptions are illustrative teaching ranges, NOT forecasts or guarantees.
"""
from __future__ import annotations

import math
from datetime import date

from src.core.models import Goal, GoalProjection, GoalScenario

DEFAULT_INFLATION = 0.025
MAX_YEARS_SEARCH = 100

# The three scenarios are centered on the user's own Goal.assumed_return_pct (0% by default: no assumed
# growth), spread by +/- SPREAD_PCT percentage points for the below/above pair (a fixed, documented
# choice, not user-configurable).
SCENARIO_LABELS = ("Below your assumption", "Your assumption", "Above your assumption")
SPREAD_PCT = 2.0


def _monthly_rate(annual_return: float) -> float:
    if annual_return <= -1:
        raise ValueError("annual_return must be greater than -100%")
    return (1 + annual_return) ** (1 / 12) - 1


def months_between(start: date, end: date) -> int:
    """Whole months from start to end (0 if end is not after start)."""
    if end <= start:
        return 0
    return max(0, round((end - start).days / 365.25 * 12))


def future_value(principal: float, monthly_contribution: float, annual_return: float, months: int) -> float:
    """Value after `months` of compounding with end-of-month contributions."""
    if principal < 0 or monthly_contribution < 0 or months < 0:
        raise ValueError("principal, contribution and months must be non-negative")
    r = _monthly_rate(annual_return)
    if months == 0:
        return principal
    growth = (1 + r) ** months
    if abs(r) < 1e-12:
        return principal + monthly_contribution * months
    return principal * growth + monthly_contribution * (growth - 1) / r


def required_monthly_contribution(target: float, principal: float, annual_return: float, months: int) -> float:
    """Monthly amount needed to reach `target`. 0 if the principal alone already gets there.

    With no time left (months == 0) the shortfall would have to be paid as a lump sum, which is returned.
    """
    if months == 0:
        return max(0.0, target - principal)
    r = _monthly_rate(annual_return)
    growth = (1 + r) ** months
    gap = target - principal * growth
    if gap <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return gap / months
    return gap * r / (growth - 1)


def years_to_reach(
    target: float, principal: float, monthly_contribution: float, annual_return: float
) -> float | None:
    """Years until `target` is reached at fixed contributions; None if not within 100 years."""
    if principal >= target:
        return 0.0
    r = _monthly_rate(annual_return)
    value = principal
    for month in range(1, MAX_YEARS_SEARCH * 12 + 1):
        value = value * (1 + r) + monthly_contribution
        if value >= target:
            return round(month / 12, 2)
    return None


def yearly_path(principal: float, monthly_contribution: float, annual_return: float, months: int) -> list[float]:
    """Balance at year 0, 1, 2, ... plus the exact end of the horizon (for charting)."""
    points = [round(principal, 2)]
    year_marks = list(range(12, months + 1, 12))
    for m in year_marks:
        points.append(round(future_value(principal, monthly_contribution, annual_return, m), 2))
    if months % 12 != 0 or months == 0:
        if months > 0:
            points.append(round(future_value(principal, monthly_contribution, annual_return, months), 2))
    return points


def _scenario(label: str, annual: float, goal: Goal, months: int, inflation: float) -> GoalScenario:
    projected = future_value(goal.current_savings, goal.monthly_contribution, annual, months)
    real = projected / ((1 + inflation) ** (months / 12)) if months else projected
    return GoalScenario(
        label=label,
        annual_return=annual,
        projected_value=round(projected, 2),
        real_projected_value=round(real, 2),
        on_track=projected >= goal.target_amount,
        shortfall=round(max(0.0, goal.target_amount - projected), 2),
        required_monthly=round(
            required_monthly_contribution(goal.target_amount, goal.current_savings, annual, months), 2
        ),
        years_to_goal=years_to_reach(goal.target_amount, goal.current_savings, goal.monthly_contribution, annual),
    )


def project_goal(
    goal: Goal,
    *,
    today: date | None = None,
    inflation: float = DEFAULT_INFLATION,
) -> GoalProjection:
    """A below/expected/above trio centered on the user's own `goal.assumed_return_pct` (0% by default:
    no assumed growth), spread by +/- SPREAD_PCT percentage points. `goal.risk_tolerance` is not consulted
    here (it still matters elsewhere, e.g. the Portfolio agent's stated-risk comparison)."""
    today = today or date.today()
    months = months_between(today, goal.target_date)
    years = round(months / 12, 2)
    mid = goal.assumed_return_pct
    low, high = (mid - SPREAD_PCT) / 100, (mid + SPREAD_PCT) / 100

    scenarios: list[GoalScenario] = []
    paths: dict[str, list[float]] = {}
    for label, annual in zip(SCENARIO_LABELS, (low, mid / 100, high)):
        scenarios.append(_scenario(label, annual, goal, months, inflation))
        paths[label] = yearly_path(goal.current_savings, goal.monthly_contribution, annual, months)
    return GoalProjection(goal=goal, years=years, scenarios=scenarios, yearly_path=paths)


def inflation_adjusted(amount: float, years: float, inflation: float = DEFAULT_INFLATION) -> float:
    """Express a future dollar amount in today's dollars."""
    if years < 0:
        raise ValueError("years must be non-negative")
    return amount / ((1 + inflation) ** years) if not math.isclose(years, 0) else amount
