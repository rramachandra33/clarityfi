"""Portfolio vs benchmark: hypothetical replay of today's holdings over past prices."""
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from src.core.freshness import DataStatus
from src.core.market_calc import compute_trend
from src.core.models import Holding, PricePoint
from src.core.portfolio_history import MIN_COMMON_DAYS, interpret, value_series
from src.data.db import Database
from src.data.market_client import History
from src.data.repositories import PortfolioRepository
from src.data.seed import seed
from src.web_app.api import main
from src.web_app.api.services import ApiServices, default_limiters


def prices(start, daily, n=120, first=date(2026, 3, 1)):
    return [PricePoint(day=first + timedelta(days=i), close=round(start * (1 + daily) ** i, 4)) for i in range(n)]


# ------------------------------------------------------------------------------------------ pure math
def test_value_series_is_shares_times_price_plus_cash_on_shared_days():
    a, b = prices(10, 0.0, 30), prices(100, 0.0, 30, first=date(2026, 3, 6))
    series, excluded, coverage = value_series([Holding(ticker="A", shares=2), Holding(ticker="B", shares=1)], {"A": a, "B": b}, cash=50)
    assert excluded == [] and coverage == 100.0 and len(series) == 25 and series[0].day == date(2026, 3, 6)
    assert all(p.close == pytest.approx(2 * 10 + 100 + 50) for p in series)


def test_missing_history_is_excluded_and_coverage_reflects_it():
    series, excluded, coverage = value_series(
        [Holding(ticker="A", shares=10), Holding(ticker="X", shares=1, cost_basis=1000)], {"A": prices(10, 0.0, 30)}, 0)
    assert excluded == ["X"] and len(series) == 30 and coverage == pytest.approx(100 / 1100 * 100, abs=0.1)


def test_nothing_usable_or_too_few_shared_days_yields_no_series():
    assert value_series([Holding(ticker="X", shares=1)], {}, 0) == ([], ["X"], 0.0)
    assert value_series([], {}, 0)[0] == []
    few = value_series([Holding(ticker="A", shares=1), Holding(ticker="B", shares=1)],
                       {"A": prices(1, 0, 30), "B": prices(1, 0, 30, first=date(2026, 3, 30))}, 0)
    assert few[0] == [] and set(few[1]) == {"A", "B"} and MIN_COMMON_DAYS >= 2
    assert value_series([Holding(ticker="A", shares=0)], {"A": prices(1, 0, 30)}, 0)[0] == []  # zero shares ignored


def test_interpretation_compares_and_caveats_without_advice():
    up, flat = compute_trend("P", prices(100, 0.004)), compute_trend("SPY", prices(100, 0.001))
    text = " ".join(interpret(up, flat, "SPY", coverage_pct=100, excluded=[]))
    assert "ahead" in text and f"{up.return_3m_pct:+.1f}%" in text and f"{flat.return_3m_pct:+.1f}%" in text
    assert "ignores when you actually bought" in text and "recommend" in text
    down = " ".join(interpret(compute_trend("P", prices(100, -0.004)), flat, "SPY", coverage_pct=80, excluded=["X"]))
    assert "behind" in down and "X could not be included" in down and "about 80%" in down and "larger drop" in down
    assert "smaller drop" in " ".join(interpret(flat, compute_trend("B", prices(100, -0.004)), "SPY", coverage_pct=100, excluded=[]))
    alone = interpret(up, None, None, coverage_pct=100, excluded=[])
    assert alone[0].endswith("%.") and "compared with" not in alone[0]
    short = interpret(compute_trend("S", prices(100, 0.01, 30)), None, None, coverage_pct=100, excluded=[])
    assert "last month" in short[0]
    steady = " ".join(interpret(compute_trend("P", prices(100, 0.001)), compute_trend("W", [PricePoint(day=date(2026, 3, 1) + timedelta(days=i), close=100 + (12 if i % 2 else -12)) for i in range(120)]), "SPY", coverage_pct=100, excluded=[]))
    assert "steadier" in steady


# -------------------------------------------------------------------------------------------- API
class Market:
    """Per-symbol histories; `status` per symbol; failing symbols raise."""

    def __init__(self, series, status=None, failing=()):
        self.series, self.status, self.failing = series, status or {}, set(failing)

    def get_trend(self, symbol, days=None):
        if symbol in self.failing or symbol not in self.series:
            raise RuntimeError("no history")
        points = self.series[symbol]
        status = self.status.get(symbol, DataStatus.live)
        hist = History(symbol, points, "mock" if status is DataStatus.sample else "yfinance", status=status, age_seconds=600)
        return compute_trend(symbol, points), hist


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "pc.db")
    db.init_schema()
    seed(db)
    market = Market({"VTI": prices(100, 0.003), "VXUS": prices(50, 0.001), "BND": prices(70, 0.0002), "SPY": prices(400, 0.002)})
    svc = ApiServices(db=db, market=market, news=None, portfolio_agent=None, goal_agent=None, news_agent=None, memory=None, limiters=default_limiters())
    main.app.dependency_overrides[main.get_services] = lambda: svc
    with TestClient(main.app) as client:
        client.market, client.db = market, db
        yield client
    main.app.dependency_overrides.clear()


def test_comparison_replays_holdings_against_the_benchmark(env):
    r = env.get("/api/portfolio/learner-105/comparison").json()
    assert r["benchmark"] == "SPY" and sorted(r["included"]) == ["BND", "VTI", "VXUS"] and r["excluded"] == {} and r["coverage_pct"] == 100.0
    assert set(r["comparison"]) == {"Your mix", "SPY"} and r["comparison"]["Your mix"][0]["value"] == 100.0
    assert r["relative"]["return_3m_diff_pts"] is not None and r["insights"][-1].startswith("This replays your current holdings")
    assert r["data_status"] == "live" and r["portfolio_trend"]["symbol"] == "Your mix"


def test_comparison_discloses_stale_history_and_never_uses_sample(env):
    env.market.status = {"VTI": DataStatus.stale, "VXUS": DataStatus.sample}
    r = env.get("/api/portfolio/learner-105/comparison").json()
    assert "VXUS" not in r["included"] and "SAMPLE" in r["excluded"]["VXUS"] and r["data_status"] == "stale"
    assert r["notices"] == ["Price history for VTI is from the last known values, which may be out of date (oldest about 10 minutes old)."]
    assert any("VXUS" in line for line in r["insights"]) and r["notices"] and r["coverage_pct"] < 100
    env.market.status = {"SPY": DataStatus.sample}
    only_holdings = env.get("/api/portfolio/learner-105/comparison").json()
    assert only_holdings["benchmark"] is None and any("SAMPLE" in n for n in only_holdings["notices"]) and only_holdings["comparison"].keys() == {"Your mix"}


def test_comparison_survives_benchmark_or_holding_failures(env):
    env.market.failing = {"SPY"}
    r = env.get("/api/portfolio/learner-105/comparison").json()
    assert r["benchmark"] is None and any("unavailable" in n for n in r["notices"]) and r["insights"]
    env.market.failing = {"VTI"}
    r2 = env.get("/api/portfolio/learner-105/comparison").json()
    assert r2["excluded"] == {"VTI": "Price history unavailable"} and "VTI" not in r2["included"]
    env.market.failing = {"VTI", "VXUS", "BND"}
    none = env.get("/api/portfolio/learner-105/comparison").json()
    assert none["portfolio_trend"] is None and any("not enough price history" in n for n in none["notices"]) and none["insights"] == []


def test_comparison_empty_unknown_and_invalid_inputs(env):
    assert env.get("/api/portfolio/learner-110/comparison").json()["empty"] is True
    assert env.get("/api/portfolio/nobody-here/comparison").status_code == 404
    assert env.get("/api/portfolio/learner-105/comparison?benchmark=%24%24").status_code == 422
    assert env.get("/api/portfolio/learner-105/comparison?days=5").status_code == 422


def test_comparison_includes_cash_and_other_benchmarks(env):
    repo = PortfolioRepository(env.db)
    p = repo.get("learner-105")
    from src.core.models import Portfolio
    repo.replace(Portfolio(user_id="learner-105", holdings=p.holdings, cash=1_000_000))
    heavy = env.get("/api/portfolio/learner-105/comparison?benchmark=vti").json()
    assert heavy["benchmark"] == "VTI" and abs(heavy["portfolio_trend"]["return_3m_pct"]) < 2  # mostly cash: barely moves


def test_several_cached_histories_produce_one_notice_naming_the_oldest(env):
    env.market.status = {"VTI": DataStatus.cached, "VXUS": DataStatus.cached, "BND": DataStatus.cached, "SPY": DataStatus.cached}
    notices = env.get("/api/portfolio/learner-105/comparison").json()["notices"]
    assert notices == ["Price history for BND, VTI and VXUS are from a saved copy rather than a fresh lookup (oldest about 10 minutes old).",
                       "Price history for SPY is from a saved copy rather than a fresh lookup (oldest about 10 minutes old)."]
