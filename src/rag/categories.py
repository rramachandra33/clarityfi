"""Category groups: agents ask for a topic ("tax"); this maps it to the KB's fine-grained categories.

The knowledge base has ~50 categories from several source files. Groups let the Tax agent (and
future filters) target a topic without hard-coding raw category names. A unit test checks that every
category present in the real KB belongs to at least one group so new content cannot silently
become unreachable.
"""
from __future__ import annotations

CATEGORY_GROUPS: dict[str, tuple[str, ...]] = {
    # Tax education also covers the account types whose rules are tax rules (Roth/traditional IRA, 401k, HSA...),
    # which the knowledge base files under "retirement".
    "tax": ("taxes_and_investing", "tax_efficiency", "tax_education", "retirement"),
    "retirement": ("retirement",),
    "stocks": ("stocks",),
    "bonds": ("bonds",),
    "funds": ("etfs_and_funds", "funds", "fund_selection", "emerging_products"),
    "portfolio": ("portfolio_management", "portfolio_construction", "optimization", "risk_measurement",
                  "risk_adjusted_performance", "dependence_measures", "limitations", "capital_market_extension"),
    "risk": ("risk", "risk_and_return", "risk_measurement", "investor_behavior_and_risk"),
    "basics": ("financial_basics", "investing_basics", "financial_foundations", "foundations", "saving_and_investing",
               "personal_finance", "planning", "cash_and_cash_equivalents", "banking", "credit_and_debt", "costs"),
    "behavior": ("behavior", "investor_behavior_and_risk", "investor_preferences", "decision_process"),
    "markets": ("market_analysis", "trading_basics", "economics", "investment_research", "investment_strategy",
                "advanced_concepts", "real_estate"),
    "protection": ("fraud_types", "fraud_prevention", "account_protection", "investor_remedies", "required_disclosures",
                   "professional_due_diligence", "professional_relationships", "standards_of_conduct", "fees_and_conflicts"),
}


def resolve_categories(topic: str | None) -> list[str] | None:
    """A group name -> its categories; a raw category name -> itself; None/'' -> None (no filter)."""
    if not topic:
        return None
    key = topic.strip().lower()
    if key in CATEGORY_GROUPS:
        return list(CATEGORY_GROUPS[key])
    return [key]


def group_of(category: str) -> list[str]:
    return [g for g, cats in CATEGORY_GROUPS.items() if category in cats]
