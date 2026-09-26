"""Deterministic synthetic data: 10 fictional users with distinct portfolios and goals.

All names, balances and holdings are invented for testing and demos; nothing here is real
personal or financial data. Seeding is idempotent and never touches non-synthetic users.

    python -m src.data.seed            # seed if the 10 synthetic users are missing
    python -m src.data.seed --reset    # rebuild only the synthetic users' portfolios/goals
"""
from __future__ import annotations

import argparse
from datetime import date
from typing import Any

from src.core.config import settings
from src.core.models import AssetClass as A, Goal, Holding, Portfolio, RiskTolerance as R, UserProfile
from src.data.db import Database
from src.data.repositories import GoalRepository, PortfolioRepository, UserRepository

# (ticker, shares, total cost basis USD, asset class, expense ratio)
H = tuple[str, float, float, A, float | None]


def _h(rows: list[H]) -> list[Holding]:
    return [Holding(ticker=t, shares=s, cost_basis=c, asset_class=a, expense_ratio=e) for t, s, c, a, e in rows]


SYNTHETIC_USERS: list[dict[str, Any]] = [
    {  # 1 - young, aggressive, growth heavy
        "profile": UserProfile(user_id="learner-101", name="Alex Rivera (demo)", risk_tolerance=R.aggressive,
                               experience_level="intermediate", horizon_years=35),
        "cash": 1500.0,
        "holdings": [("QQQ", 40, 13200, A.etf, 0.0020), ("VTI", 30, 6600, A.etf, 0.0003),
                     ("NVDA", 12, 4800, A.stock, None), ("TSLA", 10, 2400, A.stock, None)],
        "goals": [Goal(name="Down payment", target_amount=80000, target_date=date(2032, 6, 1),
                       current_savings=18000, monthly_contribution=900, risk_tolerance=R.aggressive)],
    },
    {  # 2 - near retiree, conservative income
        "profile": UserProfile(user_id="learner-102", name="Morgan Chen (demo)", risk_tolerance=R.conservative,
                               experience_level="intermediate", horizon_years=5),
        "cash": 42000.0,
        "holdings": [("BND", 900, 66000, A.bond, 0.0003), ("SCHD", 350, 26000, A.etf, 0.0006),
                     ("VTI", 120, 22000, A.etf, 0.0003), ("TIP", 300, 32000, A.bond, 0.0019)],
        "goals": [Goal(name="Retirement income", target_amount=900000, target_date=date(2031, 1, 1),
                       current_savings=610000, monthly_contribution=2500, risk_tolerance=R.conservative)],
    },
    {  # 3 - all cash saver (edge case: no invested holdings)
        "profile": UserProfile(user_id="learner-103", name="Jordan Patel (demo)", risk_tolerance=R.conservative,
                               experience_level="beginner", horizon_years=8),
        "cash": 24000.0,
        "holdings": [],
        "goals": [Goal(name="Emergency fund", target_amount=30000, target_date=date(2027, 12, 1),
                       current_savings=24000, monthly_contribution=500, risk_tolerance=R.conservative)],
    },
    {  # 4 - concentrated in one stock (edge case: high concentration risk)
        "profile": UserProfile(user_id="learner-104", name="Sam Okafor (demo)", risk_tolerance=R.aggressive,
                               experience_level="beginner", horizon_years=20),
        "cash": 800.0,
        "holdings": [("AAPL", 220, 26000, A.stock, None), ("VOO", 4, 1600, A.etf, 0.0003)],
        "goals": [Goal(name="Financial independence", target_amount=1500000, target_date=date(2046, 1, 1),
                       current_savings=28000, monthly_contribution=700, risk_tolerance=R.aggressive)],
    },
    {  # 5 - three-fund index investor
        "profile": UserProfile(user_id="learner-105", name="Riley Nguyen (demo)", risk_tolerance=R.moderate,
                               experience_level="advanced", horizon_years=25),
        "cash": 3000.0,
        "holdings": [("VTI", 260, 52000, A.etf, 0.0003), ("VXUS", 300, 17500, A.etf, 0.0007),
                     ("BND", 180, 13000, A.bond, 0.0003)],
        "goals": [Goal(name="Retirement", target_amount=1800000, target_date=date(2050, 1, 1),
                       current_savings=86000, monthly_contribution=1800, risk_tolerance=R.moderate)],
    },
    {  # 6 - bond heavy
        "profile": UserProfile(user_id="learner-106", name="Casey Brooks (demo)", risk_tolerance=R.conservative,
                               experience_level="beginner", horizon_years=12),
        "cash": 6000.0,
        "holdings": [("AGG", 400, 41000, A.bond, 0.0003), ("VGSH", 300, 17500, A.bond, 0.0004),
                     ("VTI", 25, 5200, A.etf, 0.0003)],
        "goals": [Goal(name="Child's education", target_amount=120000, target_date=date(2036, 8, 1),
                       current_savings=64000, monthly_contribution=600, risk_tolerance=R.conservative)],
    },
    {  # 7 - crypto tilt
        "profile": UserProfile(user_id="learner-107", name="Devon Kim (demo)", risk_tolerance=R.aggressive,
                               experience_level="intermediate", horizon_years=15),
        "cash": 1200.0,
        "holdings": [("BTC-USD", 0.35, 14000, A.crypto, None), ("ETH-USD", 3.0, 6000, A.crypto, None),
                     ("VTI", 45, 9500, A.etf, 0.0003), ("COIN", 15, 2700, A.stock, None)],
        "goals": [Goal(name="Travel fund", target_amount=25000, target_date=date(2029, 5, 1),
                       current_savings=9000, monthly_contribution=400, risk_tolerance=R.aggressive)],
    },
    {  # 8 - balanced mid-career, two goals
        "profile": UserProfile(user_id="learner-108", name="Taylor Singh (demo)", risk_tolerance=R.moderate,
                               experience_level="intermediate", horizon_years=22),
        "cash": 9000.0,
        "holdings": [("VOO", 110, 38000, A.etf, 0.0003), ("VXUS", 150, 8800, A.etf, 0.0007),
                     ("BND", 200, 14500, A.bond, 0.0003), ("MSFT", 20, 6400, A.stock, None)],
        "goals": [Goal(name="Home renovation", target_amount=45000, target_date=date(2028, 9, 1),
                       current_savings=15000, monthly_contribution=800, risk_tolerance=R.moderate),
                  Goal(name="Retirement", target_amount=1400000, target_date=date(2048, 1, 1),
                       current_savings=72000, monthly_contribution=1500, risk_tolerance=R.moderate)],
    },
    {  # 9 - real estate / REIT tilt
        "profile": UserProfile(user_id="learner-109", name="Jamie Lopez (demo)", risk_tolerance=R.moderate,
                               experience_level="intermediate", horizon_years=18),
        "cash": 5000.0,
        "holdings": [("VNQ", 260, 22000, A.real_estate, 0.0013), ("O", 150, 8500, A.real_estate, None),
                     ("VTI", 70, 14500, A.etf, 0.0003), ("BND", 90, 6600, A.bond, 0.0003)],
        "goals": [Goal(name="Rental property deposit", target_amount=100000, target_date=date(2033, 3, 1),
                       current_savings=40000, monthly_contribution=1100, risk_tolerance=R.moderate)],
    },
    {  # 10 - brand-new investor, no holdings/cash (edge case: empty portfolio)
        "profile": UserProfile(user_id="learner-110", name="Avery Walker (demo)", risk_tolerance=R.moderate,
                               experience_level="beginner", horizon_years=30),
        "cash": 0.0,
        "holdings": [],
        "goals": [],
    },
]


def seed(db: Database, *, reset: bool = False) -> int:
    """Insert (or, with reset, rebuild) the synthetic users. Returns how many users were written."""
    users, portfolios, goals = UserRepository(db), PortfolioRepository(db), GoalRepository(db)
    written = 0
    for spec in SYNTHETIC_USERS:
        profile: UserProfile = spec["profile"]
        exists = users.get(profile.user_id) is not None
        if exists and not reset:
            continue
        users.upsert_synthetic(profile)
        portfolios.replace(Portfolio(user_id=profile.user_id, holdings=_h(spec["holdings"]), cash=spec["cash"]))
        goals.delete_all(profile.user_id)
        for goal in spec["goals"]:
            goals.add(profile.user_id, goal)
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the 10 synthetic ClarityFI users.")
    parser.add_argument("--reset", action="store_true", help="Rebuild synthetic users' portfolios and goals")
    args = parser.parse_args()
    db = Database(settings.app_db_path)
    db.init_schema()
    print(f"Seeded {seed(db, reset=args.reset)} synthetic users into {db.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
