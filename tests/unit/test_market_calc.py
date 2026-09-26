import math
from datetime import date, timedelta

import pytest

from src.core.market_calc import (
    annualized_volatility_pct, compute_trend, max_drawdown_pct, pct_change, sma, trend_label,
)
from src.core.models import PricePoint


def series(values, start=date(2026, 1, 1)):
    return [PricePoint(day=start + timedelta(days=i), close=v) for i, v in enumerate(values)]


def test_sma_and_pct_change():
    assert sma([1, 2, 3, 4], 2) == 3.5 and sma([1], 2) is None
    assert pct_change(110, 100) == 10.0 and pct_change(90, 100) == -10.0 and pct_change(5, 0) is None


def test_max_drawdown():
    assert max_drawdown_pct([100, 120, 90, 110]) == -25.0
    assert max_drawdown_pct([1, 2, 3]) == 0.0


def test_volatility_needs_enough_points_and_matches_definition():
    assert annualized_volatility_pct([1, 2, 3]) is None
    flat = annualized_volatility_pct([100.0] * 30)
    assert flat == 0.0
    values = [100 * (1.01 if i % 2 else 0.99) ** i for i in range(30)]
    logs = [math.log(b / a) for a, b in zip(values, values[1:])]
    mean = sum(logs) / len(logs)
    expected = math.sqrt(sum((x - mean) ** 2 for x in logs) / (len(logs) - 1)) * math.sqrt(252) * 100
    assert annualized_volatility_pct(values) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("last,s20,s50,label", [
    (110, 105, 100, "Uptrend"), (90, 95, 100, "Downtrend"), (100, 105, 100, "Sideways / mixed"),
    (110, None, None, "Insufficient history"), (110, 100, None, "Above short-term average"),
    (90, 100, None, "Below short-term average"),
])
def test_trend_labels(last, s20, s50, label):
    assert trend_label(last, s20, s50) == label


def test_steady_uptrend_summary():
    t = compute_trend("VTI", series([100 + i * 0.5 for i in range(120)]))
    assert t.trend_label == "Uptrend" and t.max_drawdown_pct == 0.0
    assert t.return_1m_pct > 0 and t.return_3m_pct > t.return_1m_pct
    assert t.period_high == t.last_close and t.period_low == 100.0 and t.points == 120
    assert t.sma_20 > t.sma_50 and any("past prices" in n for n in t.notes)


def test_downtrend_with_large_drawdown_and_volatility_notes():
    t = compute_trend("XYZ", series([200 - i * 1.2 for i in range(100)]))
    assert t.trend_label == "Downtrend" and t.max_drawdown_pct < -20
    assert any("fell as much as" in n for n in t.notes)


def test_short_history_omits_long_windows():
    t = compute_trend("NEW", series([10, 11, 12, 11, 12, 13, 12, 13, 14, 13, 14]))
    assert t.sma_20 is None and t.sma_50 is None and t.return_1m_pct is None
    assert t.trend_label == "Insufficient history" and any("Limited history" in n for n in t.notes)


def test_input_is_sorted_and_needs_two_points():
    unordered = series([1, 2, 3, 4, 5])[::-1]
    assert compute_trend("A", unordered).last_close == 5
    with pytest.raises(ValueError):
        compute_trend("A", series([1]))
    with pytest.raises(ValueError):
        compute_trend("A", [])


def test_volatile_and_steady_descriptors():
    wild = compute_trend("W", series([100 * (1.06 if i % 2 else 0.94) for i in range(40)]))
    assert any("very volatile" in n for n in wild.notes)
    calm = compute_trend("C", series([100 + (0.05 if i % 2 else -0.05) for i in range(40)]))
    assert any("relatively steady" in n for n in calm.notes)
