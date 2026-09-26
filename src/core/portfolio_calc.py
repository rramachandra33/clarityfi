"""Deterministic portfolio analytics (no LLM, no network).

Prices are passed in (from the market-data layer or a test fixture), so every function is pure.
All outputs are educational descriptors, never buy/sell recommendations.

Method notes (also documented in docs/design.md):
- Values: shares x price. If a price is missing we fall back to cost basis, else 0, and say so.
- Diversification score (0-100) = 45% effective-position count + 35% asset-class spread
  + 20% single-position concentration. Broad index funds are treated as internally
  diversified via a look-through factor (each counts as ~20 positions).
- Risk score (0-100) = value-weighted asset-class risk weights, plus a concentration add-on.
"""
from __future__ import annotations

from datetime import datetime
from typing import Mapping

from src.core.models import AssetClass, Portfolio, PortfolioMetrics, RiskTolerance

# Approximate look-through: broad, many-holding funds. Everything else counts as one position.
BROAD_FUNDS: frozenset[str] = frozenset(
    {"VTI", "VOO", "SPY", "IVV", "ITOT", "SCHB", "VT", "VXUS", "IXUS", "VEA", "VWO", "VTV", "VUG",
     "BND", "AGG", "BNDX", "SCHZ", "TIP", "VGSH", "VGIT", "VGLT", "VNQ", "SCHD", "QQQ", "IWM", "DIA"}
)
BROAD_FUND_LOOKTHROUGH = 20

# 0 = no market risk, 1 = highest. Coarse by design; used only to label a mix, not to predict.
ASSET_CLASS_RISK: dict[AssetClass, float] = {
    AssetClass.cash: 0.0,
    AssetClass.bond: 0.25,
    AssetClass.real_estate: 0.70,
    AssetClass.etf: 0.80,
    AssetClass.mutual_fund: 0.80,
    AssetClass.stock: 0.90,
    AssetClass.other: 0.70,
    AssetClass.crypto: 1.00,
}
_FUND_CLASSES = {AssetClass.etf, AssetClass.mutual_fund, AssetClass.bond, AssetClass.real_estate}
_TOLERANCE_RANK = {RiskTolerance.conservative: 0, RiskTolerance.moderate: 1, RiskTolerance.aggressive: 2}
_LABEL_RANK = {"Conservative": 0, "Moderate": 1, "Growth": 2, "Aggressive": 2}

TARGET_EFFECTIVE_POSITIONS = 6
TARGET_EFFECTIVE_CLASSES = 4


def risk_label_for(score: float) -> str:
    if score < 30:
        return "Conservative"
    if score < 60:
        return "Moderate"
    if score < 80:
        return "Growth"
    return "Aggressive"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round(value: float, places: int = 2) -> float:
    return round(value + 0.0, places)  # +0.0 normalizes -0.0


def _hhi(weights: list[float]) -> float:
    return sum(w * w for w in weights)


def compute_portfolio_metrics(
    portfolio: Portfolio,
    prices: Mapping[str, float] | None = None,
    *,
    stated_risk: RiskTolerance | None = None,
    as_of: datetime | None = None,
) -> PortfolioMetrics:
    """Analyze a portfolio. `prices` maps ticker -> latest price (USD)."""
    prices = {k.upper(): v for k, v in (prices or {}).items() if v is not None and v >= 0}
    notes: list[str] = []

    values: dict[str, float] = {}
    classes: dict[AssetClass, float] = {}
    costs: list[float] = []
    fee_num = fee_den = 0.0
    unknown_fee_funds: list[str] = []
    fallbacks: list[str] = []
    missing: list[str] = []

    for h in portfolio.holdings:
        price = prices.get(h.ticker)
        if price is not None:
            value = h.shares * price
        elif h.cost_basis is not None:
            value = h.cost_basis
            fallbacks.append(h.ticker)
        else:
            value = 0.0
            missing.append(h.ticker)
        values[h.ticker] = values.get(h.ticker, 0.0) + value
        classes[h.asset_class] = classes.get(h.asset_class, 0.0) + value
        if h.cost_basis is not None:
            costs.append(h.cost_basis)

        if h.expense_ratio is not None:
            fee_num += value * h.expense_ratio
            fee_den += value
        elif h.asset_class in _FUND_CLASSES:
            unknown_fee_funds.append(h.ticker)
        else:  # stocks/crypto/other carry no fund expense ratio
            fee_den += value

    invested = sum(values.values())
    cash = portfolio.cash
    total = invested + cash

    if fallbacks:
        notes.append(f"No live price for {', '.join(sorted(set(fallbacks)))}; cost basis was used as its value.")
    if missing:
        notes.append(f"No price or cost basis for {', '.join(sorted(set(missing)))}; valued at $0.")
    if unknown_fee_funds:
        notes.append(
            f"Expense ratio unknown for {', '.join(sorted(set(unknown_fee_funds)))}; excluded from the fee estimate."
        )

    if total <= 0:
        return PortfolioMetrics(
            total_value=0.0,
            diversification_score=0.0,
            risk_label="Not enough data",
            notes=notes + ["This portfolio is empty. Add holdings or cash to see an analysis."],
            **({"as_of": as_of} if as_of else {}),
        )

    if cash > 0:
        classes[AssetClass.cash] = classes.get(AssetClass.cash, 0.0) + cash

    by_holding = {t: _round(v / total * 100) for t, v in sorted(values.items(), key=lambda kv: -kv[1])}
    if cash > 0:
        by_holding["CASH"] = _round(cash / total * 100)
    by_class = {c.value: _round(v / total * 100) for c, v in sorted(classes.items(), key=lambda kv: -kv[1])}

    # --- gain / loss (only when every priced holding has a cost basis) -----------------------
    total_cost = unrealized = gain_pct = None
    if portfolio.holdings and len(costs) == len(portfolio.holdings) and sum(costs) > 0:
        total_cost = sum(costs)
        unrealized = invested - total_cost
        gain_pct = _round(unrealized / total_cost * 100)
        unrealized = _round(unrealized)

    # --- fees --------------------------------------------------------------------------------
    weighted_er = annual_fee = None
    if fee_den > 0 and fee_num >= 0 and any(h.expense_ratio is not None for h in portfolio.holdings):
        weighted_er = round(fee_num / fee_den * 100, 4)  # percent
        annual_fee = _round(fee_num)
        if weighted_er > 0.5:
            notes.append(f"Weighted expense ratio is {weighted_er:.2f}%; fees compound over time, so it is worth comparing costs.")

    # --- diversification ---------------------------------------------------------------------
    eff_positions = 0.0
    top_single = 0.0
    if invested > 0:
        w = {t: v / invested for t, v in values.items()}
        denom = sum((wi**2) / (BROAD_FUND_LOOKTHROUGH if t in BROAD_FUNDS else 1) for t, wi in w.items())
        eff_positions = 1 / denom if denom > 0 else 0.0
        top_single = max((wi for t, wi in w.items() if t not in BROAD_FUNDS), default=0.0)
        top_single_of_total = top_single * invested / total
    else:
        top_single_of_total = 0.0

    class_weights = [v / total for v in classes.values() if v > 0]
    eff_classes = 1 / _hhi(class_weights) if class_weights else 0.0

    position_component = _clamp((eff_positions - 1) / (TARGET_EFFECTIVE_POSITIONS - 1)) if invested > 0 else 0.0
    class_component = _clamp((eff_classes - 1) / (TARGET_EFFECTIVE_CLASSES - 1))
    concentration_component = 1.0 - _clamp((top_single_of_total - 0.10) / 0.50)
    if invested <= 0:
        concentration_component = 0.0
    diversification = _round(100 * (0.45 * position_component + 0.35 * class_component + 0.20 * concentration_component), 1)

    # --- risk --------------------------------------------------------------------------------
    risk_index = sum(ASSET_CLASS_RISK[c] * v for c, v in classes.items()) / total
    risk_score = risk_index * 100
    if top_single_of_total > 0.40:
        risk_score = min(100.0, risk_score + 10)
    risk_score = _round(risk_score, 1)
    label = risk_label_for(risk_score)

    # --- educational observations ------------------------------------------------------------
    if invested > 0:
        biggest = max(((t, wt) for t, wt in by_holding.items() if t not in BROAD_FUNDS and t != "CASH"),
                      key=lambda kv: kv[1], default=None)
        if biggest and biggest[1] >= 25:
            notes.append(
                f"{biggest[0]} makes up {biggest[1]:.0f}% of the portfolio. Heavy concentration in one holding "
                "means its ups and downs drive overall results."
            )
        if len(values) == 1 and next(iter(values)) not in BROAD_FUNDS:
            notes.append("The portfolio holds a single position; spreading across holdings is a common way to reduce single-company risk.")
    if cash / total >= 0.30 and invested > 0:
        notes.append(f"Cash is {cash / total * 100:.0f}% of the portfolio; cash is stable but may not keep pace with inflation over long horizons.")
    if invested == 0:
        notes.append("Everything is held in cash: very stable, but with limited long-term growth potential.")
    if by_class.get(AssetClass.crypto.value, 0) >= 20:
        notes.append("A large crypto share adds substantial volatility compared with diversified funds.")
    if stated_risk is not None:
        gap = _LABEL_RANK[label] - _TOLERANCE_RANK[stated_risk]
        if gap > 0:
            notes.append(f"The mix looks more aggressive ({label}) than the stated {stated_risk.value} comfort level.")
        elif gap < 0:
            notes.append(f"The mix looks more cautious ({label}) than the stated {stated_risk.value} comfort level, which may lower long-term growth potential.")

    kwargs = {"as_of": as_of} if as_of else {}
    return PortfolioMetrics(
        total_value=_round(total),
        invested_value=_round(invested),
        cash_value=_round(cash),
        allocation_by_asset_class=by_class,
        allocation_by_holding=by_holding,
        total_cost_basis=_round(total_cost) if total_cost is not None else None,
        unrealized_gain_loss=unrealized,
        unrealized_gain_pct=gain_pct,
        weighted_expense_ratio=weighted_er,
        annual_fee_estimate=annual_fee,
        effective_positions=_round(eff_positions),
        diversification_score=_clamp(diversification, 0, 100),
        risk_score=_clamp(risk_score, 0, 100),
        risk_label=label,
        notes=notes,
        **kwargs,
    )
