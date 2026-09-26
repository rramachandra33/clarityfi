import pytest

from src.core.models import AssetClass as A, Holding, Portfolio, RiskTolerance as R
from src.core.portfolio_calc import compute_portfolio_metrics, risk_label_for
from src.data.db import Database
from src.data.repositories import PortfolioRepository
from src.data.seed import SYNTHETIC_USERS, seed


def pf(holdings, cash=0.0):
    return Portfolio(user_id="tester-1", holdings=holdings, cash=cash)


def h(ticker, shares, cls=A.stock, cost=None, er=None):
    return Holding(ticker=ticker, shares=shares, cost_basis=cost, asset_class=cls, expense_ratio=er)


def test_empty_portfolio_is_handled():
    m = compute_portfolio_metrics(pf([]))
    assert m.total_value == 0 and m.diversification_score == 0 and m.risk_label == "Not enough data"
    assert any("empty" in n.lower() for n in m.notes)


def test_values_and_allocations():
    m = compute_portfolio_metrics(pf([h("VTI", 10, A.etf)], cash=1000), {"VTI": 200})
    assert m.total_value == 3000 and m.invested_value == 2000 and m.cash_value == 1000
    assert m.allocation_by_asset_class == {"etf": 66.67, "cash": 33.33}
    assert m.allocation_by_holding == {"VTI": 66.67, "CASH": 33.33}


def test_price_lookup_is_case_insensitive_and_ignores_negative_prices():
    m = compute_portfolio_metrics(pf([h("VTI", 1, A.etf, cost=90)]), {"vti": -5})
    assert m.total_value == 90  # negative price ignored -> cost basis fallback
    assert compute_portfolio_metrics(pf([h("VTI", 1, A.etf)]), {"vti": 100}).total_value == 100


def test_missing_price_falls_back_to_cost_basis_with_note():
    m = compute_portfolio_metrics(pf([h("XYZ", 5, cost=500)]), {})
    assert m.total_value == 500 and any("cost basis was used" in n for n in m.notes)


def test_missing_price_and_cost_is_valued_at_zero_with_note():
    m = compute_portfolio_metrics(pf([h("XYZ", 5)], cash=100), {})
    assert m.total_value == 100 and any("valued at $0" in n for n in m.notes)


def test_unrealized_gain_loss():
    m = compute_portfolio_metrics(pf([h("VTI", 10, A.etf, cost=1500)]), {"VTI": 200})
    assert m.total_cost_basis == 1500 and m.unrealized_gain_loss == 500 and m.unrealized_gain_pct == 33.33
    loss = compute_portfolio_metrics(pf([h("VTI", 10, A.etf, cost=2500)]), {"VTI": 200})
    assert loss.unrealized_gain_loss == -500 and loss.unrealized_gain_pct == -20.0


def test_gain_is_omitted_when_any_cost_basis_missing():
    m = compute_portfolio_metrics(pf([h("VTI", 1, A.etf, cost=100), h("QQQ", 1, A.etf)]), {"VTI": 1, "QQQ": 1})
    assert m.unrealized_gain_loss is None


def test_weighted_expense_ratio_is_value_weighted_percent():
    m = compute_portfolio_metrics(
        pf([h("VTI", 10, A.etf, er=0.0003), h("QQQ", 10, A.etf, er=0.002)]), {"VTI": 100, "QQQ": 100}
    )
    assert m.weighted_expense_ratio == pytest.approx(0.115, abs=1e-4)
    assert m.annual_fee_estimate == pytest.approx(2.3, abs=0.01)


def test_individual_stocks_count_as_zero_fee_but_unknown_fund_fees_are_excluded():
    m = compute_portfolio_metrics(
        pf([h("AAPL", 10), h("VTI", 10, A.etf, er=0.001), h("MYST", 10, A.etf)]),
        {"AAPL": 100, "VTI": 100, "MYST": 100},
    )
    # AAPL (0%) and VTI (0.1%) equal weight -> 0.05%; the fund with unknown ER is excluded and flagged
    assert m.weighted_expense_ratio == pytest.approx(0.05, abs=1e-4)
    assert any("MYST" in n and "Expense ratio unknown" in n for n in m.notes)


def test_no_fee_info_gives_none():
    assert compute_portfolio_metrics(pf([h("AAPL", 1)]), {"AAPL": 10}).weighted_expense_ratio is None


def test_high_fees_get_a_note():
    m = compute_portfolio_metrics(pf([h("PRICEY", 10, A.mutual_fund, er=0.012)]), {"PRICEY": 100})
    assert any("expense ratio" in n.lower() for n in m.notes)


def test_single_stock_is_concentrated_and_flagged():
    m = compute_portfolio_metrics(pf([h("AAPL", 100)]), {"AAPL": 100})
    assert m.diversification_score < 15
    assert any("AAPL makes up 100%" in n for n in m.notes) and any("single position" in n for n in m.notes)


def test_three_fund_portfolio_scores_much_higher_than_single_stock():
    three = compute_portfolio_metrics(
        pf([h("VTI", 60, A.etf), h("VXUS", 30, A.etf), h("BND", 40, A.bond)], cash=1000),
        {"VTI": 100, "VXUS": 100, "BND": 100},
    )
    single = compute_portfolio_metrics(pf([h("AAPL", 100)]), {"AAPL": 100})
    assert three.diversification_score > 65 > single.diversification_score
    assert three.effective_positions > single.effective_positions


def test_diversification_improves_when_adding_holdings():
    prices = {"A": 100, "B": 100, "C": 100, "D": 100}
    scores = []
    for n in (1, 2, 3, 4):
        holdings = [h(t, 10) for t in "ABCD"[:n]]
        scores.append(compute_portfolio_metrics(pf(holdings), prices).diversification_score)
    assert scores == sorted(scores) and scores[0] < scores[-1]


@pytest.mark.parametrize(
    "holdings,cash,expected",
    [
        ([], 5000, "Conservative"),
        ([h("BND", 100, A.bond)], 0, "Conservative"),
        ([h("BTC-USD", 1, A.crypto)], 0, "Aggressive"),
        ([h("VTI", 10, A.etf), h("BND", 10, A.bond)], 0, "Moderate"),
    ],
)
def test_risk_labels(holdings, cash, expected):
    m = compute_portfolio_metrics(pf(holdings, cash), {"BND": 100, "BTC-USD": 100, "VTI": 100})
    assert m.risk_label == expected


def test_risk_label_thresholds():
    assert [risk_label_for(s) for s in (0, 29.9, 30, 59.9, 60, 79.9, 80, 100)] == [
        "Conservative", "Conservative", "Moderate", "Moderate", "Growth", "Growth", "Aggressive", "Aggressive"]


def test_stated_risk_mismatch_notes():
    aggressive_mix = pf([h("BTC-USD", 1, A.crypto)])
    up = compute_portfolio_metrics(aggressive_mix, {"BTC-USD": 100}, stated_risk=R.conservative)
    assert any("more aggressive" in n for n in up.notes)
    cautious_mix = pf([], cash=5000)
    down = compute_portfolio_metrics(cautious_mix, stated_risk=R.aggressive)
    assert any("more cautious" in n for n in down.notes)
    same = compute_portfolio_metrics(cautious_mix, stated_risk=R.conservative)
    assert not any("comfort level" in n for n in same.notes)


def test_cash_heavy_and_crypto_notes():
    m = compute_portfolio_metrics(pf([h("VTI", 1, A.etf)], cash=900), {"VTI": 100})
    assert any("Cash is 90%" in n for n in m.notes)
    c = compute_portfolio_metrics(pf([h("BTC-USD", 1, A.crypto), h("VTI", 1, A.etf)]), {"BTC-USD": 100, "VTI": 100})
    assert any("crypto" in n.lower() for n in c.notes)


def test_duplicate_tickers_are_aggregated():
    m = compute_portfolio_metrics(pf([h("VTI", 1, A.etf), h("vti", 2, A.etf)]), {"VTI": 100})
    assert m.total_value == 300 and list(m.allocation_by_holding) == ["VTI"]


def test_notes_never_use_directive_language():
    for spec in SYNTHETIC_USERS:
        m = compute_portfolio_metrics(pf([h(t, s, a, c, e) for t, s, c, a, e in spec["holdings"]], spec["cash"]),
                                      {t: c / s for t, s, c, *_ in spec["holdings"]})
        text = " ".join(m.notes).lower()
        assert not any(bad in text for bad in ("you should buy", "you should sell", "sell your", "buy more"))


@pytest.fixture(scope="module")
def synthetic_metrics(tmp_path_factory):
    db = Database(tmp_path_factory.mktemp("calc") / "s.db")
    db.init_schema()
    seed(db)
    repo, out = PortfolioRepository(db), {}
    for spec in SYNTHETIC_USERS:
        uid = spec["profile"].user_id
        portfolio = repo.get(uid)
        prices = {x.ticker: x.cost_basis / x.shares for x in portfolio.holdings}  # price == avg cost
        out[uid] = compute_portfolio_metrics(portfolio, prices, stated_risk=spec["profile"].risk_tolerance)
    return out


def test_all_synthetic_users_produce_sane_metrics(synthetic_metrics):
    assert len(synthetic_metrics) == 10
    for uid, m in synthetic_metrics.items():
        assert 0 <= m.diversification_score <= 100 and 0 <= m.risk_score <= 100, uid
        if m.total_value > 0:
            assert sum(m.allocation_by_asset_class.values()) == pytest.approx(100, abs=0.05), uid
            assert sum(m.allocation_by_holding.values()) == pytest.approx(100, abs=0.05), uid


def test_synthetic_profiles_rank_as_designed(synthetic_metrics):
    m = synthetic_metrics
    assert m["learner-110"].risk_label == "Not enough data"
    assert m["learner-103"].risk_label == "Conservative"  # all cash
    assert m["learner-105"].diversification_score > m["learner-104"].diversification_score  # 3-fund vs single stock
    assert m["learner-106"].risk_score < m["learner-107"].risk_score  # bond heavy vs crypto tilt
    assert m["learner-104"].effective_positions < m["learner-105"].effective_positions
    assert any("AAPL" in n for n in m["learner-104"].notes)
