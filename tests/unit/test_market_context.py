"""Market context: benchmark comparison, rebased series, plain-language interpretation, segment overview."""
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from src.core.freshness import DataStatus
from src.core.market_calc import SEGMENTS, compute_trend, interpret, rebased, relative_performance
from src.core.models import PricePoint
from src.data.db import Database
from src.data.seed import seed
from src.web_app.api import main
from src.web_app.api.routes_market import _takeaways
from src.web_app.api.models import SegmentOut
from src.web_app.api.services import ApiServices, default_limiters
from tests.unit._agent_fakes import FakeMarket, quote


def series(start_price, daily, n=120, first=date(2026, 3, 1)):
    return [PricePoint(day=first + timedelta(days=i), close=start_price * (1 + daily) ** i) for i in range(n)]


UP, DOWN, FLAT = compute_trend("UP", series(100, 0.004)), compute_trend("DOWN", series(100, -0.004)), compute_trend("FLAT", series(100, 0.0))


def test_rebased_uses_shared_dates_and_starts_at_100():
    a, b = series(50, 0.01, 100), series(200, 0.002, 90, first=date(2026, 3, 11))
    out = rebased({"A": a, "B": b})
    assert out["A"][0]["value"] == out["B"][0]["value"] == 100.0
    assert [p["day"] for p in out["A"]] == [p["day"] for p in out["B"]] and len(out["A"]) == 90
    assert out["A"][-1]["value"] > out["B"][-1]["value"]
    assert rebased({}) == {} and rebased({"A": a, "B": series(1, 0.0, 5, first=date(2030, 1, 1))}) == {}  # no overlap
    assert rebased({"A": [PricePoint(day=date(2026, 1, 1), close=0.0), PricePoint(day=date(2026, 1, 2), close=1.0)]}) == {}


def test_relative_performance_is_percentage_point_gaps():
    rel = relative_performance(UP, DOWN)
    assert rel["return_3m_diff_pts"] == pytest.approx(UP.return_3m_pct - DOWN.return_3m_pct, abs=0.01)
    assert relative_performance(compute_trend("S", series(1, 0.01, 5)), UP)["return_3m_diff_pts"] is None


def test_interpretation_describes_pattern_using_only_computed_values():
    up = interpret(UP)
    assert "uptrend" in up[0].lower() and any("highest close" in s for s in up) and up[-1].startswith("This describes the past only")
    down = interpret(DOWN)
    assert "downtrend" in down[0].lower() and any("below its highest close" in s for s in down)
    mixed = interpret(compute_trend("M", series(100, 0.01, 60) + series(170, -0.03, 5, first=date(2026, 5, 1))))
    assert "mixed" in mixed[0].lower()
    joined = " ".join(interpret(UP, DOWN, "SPY"))
    assert "ahead of SPY" in joined and f"{UP.return_3m_pct:+.1f}%" in joined and f"{DOWN.return_3m_pct:+.1f}%" in joined
    assert "behind" in " ".join(interpret(DOWN, UP, "SPY"))
    assert not any(w in joined.lower() for w in ("you should", "buy", "sell", "will rise", "will fall"))


def test_interpretation_handles_short_history_and_volatility_gaps():
    short = compute_trend("S", series(100, 0.01, 15))
    assert interpret(short)[-1].startswith("This describes")  # no crash without averages
    wild = compute_trend("W", [PricePoint(day=date(2026, 3, 1) + timedelta(days=i), close=100 + (15 if i % 2 else -15)) for i in range(60)])
    text = " ".join(interpret(wild, FLAT, "SPY"))
    assert "very high" in text or "high" in text and "more volatile than SPY" in text


# ---------------------------------------------------------------------------------------------- API
@pytest.fixture
def client(tmp_path):
    db = Database(tmp_path / "c.db")
    db.init_schema()
    seed(db)
    market = FakeMarket({"AAPL": quote("AAPL", 200.0)})
    svc = ApiServices(db=db, market=market, news=None, portfolio_agent=None, goal_agent=None, news_agent=None, memory=None, limiters=default_limiters())
    main.app.dependency_overrides[main.get_services] = lambda: svc
    with TestClient(main.app) as c:
        c.market = market
        yield c
    main.app.dependency_overrides.clear()


def test_context_compares_with_the_default_benchmark(client):
    r = client.get("/api/market/context/aapl").json()
    assert r["symbol"] == "AAPL" and r["benchmark"] == "SPY" and r["insights"] and r["insights"][-1].startswith("This describes")
    assert set(r["comparison"]) == {"AAPL", "SPY"} and r["comparison"]["AAPL"][0]["value"] == 100.0
    assert r["relative"]["return_3m_diff_pts"] is not None and client.market.trend_calls == ["AAPL", "SPY"]


def test_context_same_symbol_and_bad_symbols_and_benchmark_failure(client):
    same = client.get("/api/market/context/SPY").json()
    assert same["benchmark"] is None and list(same["comparison"]) == ["SPY"]
    assert client.get("/api/market/context/$$$").status_code == 422
    assert client.get("/api/market/context/AAPL?benchmark=%24%24").status_code == 422
    orig = client.market.get_trend

    def flaky(symbol, days=None):
        if symbol == "SPY":
            raise RuntimeError("down")
        return orig(symbol, days)
    client.market.get_trend = flaky
    degraded = client.get("/api/market/context/AAPL").json()
    assert degraded["benchmark"] is None and any("unavailable" in n for n in degraded["notices"]) and degraded["insights"]


def test_context_discloses_non_live_history(client):
    client.market.trend_status = DataStatus.stale
    r = client.get("/api/market/context/AAPL").json()
    assert r["data_status"] == "stale" and r["notices"]


def test_overview_lists_every_segment_with_takeaways_and_disclosure(client):
    r = client.get("/api/market/overview").json()
    assert [s["symbol"] for s in r["segments"]] == [s[0] for s in SEGMENTS] and all(s["return_3m_pct"] is not None for s in r["segments"])
    assert r["takeaways"] and r["takeaways"][-1].startswith("Segments are shown through ETF proxies")
    client.market.trend_status = DataStatus.stale
    stale = client.get("/api/market/overview").json()
    assert stale["data_status"] == "stale" and stale["notices"] == ["SPY, QQQ, IWM, VXUS and BND are the last known values, which may be out of date."]


def test_overview_survives_a_failing_segment(client):
    orig = client.market.get_trend

    def flaky(symbol, days=None):
        if symbol == "IWM":
            raise RuntimeError("down")
        return orig(symbol, days)
    client.market.get_trend = flaky
    r = client.get("/api/market/overview").json()
    bad = next(s for s in r["segments"] if s["symbol"] == "IWM")
    assert bad["error"] and any("IWM" in n or "small" in n.lower() for n in r["notices"]) and len(r["takeaways"]) >= 1


def test_takeaways_compare_stocks_and_bonds():
    def seg(sym, label, r3):
        return SegmentOut(symbol=sym, label=label, description="", return_3m_pct=r3)
    opp = _takeaways([seg("SPY", "US large companies", 6.0), seg("BND", "US bonds", -1.0)])
    assert "US large companies led (+6.0%)" in opp[0] and "US bonds lagged (-1.0%)" in opp[0] and "opposite directions" in opp[1]
    same = _takeaways([seg("SPY", "US large companies", 6.0), seg("BND", "US bonds", 1.0)])
    assert "same direction" in same[1]
    assert _takeaways([seg("SPY", "US large companies", 6.0)]) == []
    one_month = _takeaways([SegmentOut(symbol="SPY", label="A", description="", return_1m_pct=2.0), SegmentOut(symbol="QQQ", label="B", description="", return_1m_pct=1.0)])
    assert "1 month" in one_month[0]
