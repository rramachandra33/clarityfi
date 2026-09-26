"""Shared domain models used by agents, services, the workflow, and the API."""
from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from src.core.freshness import DataStatus, build_notice


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RiskTolerance(str, Enum):
    conservative = "conservative"
    moderate = "moderate"
    aggressive = "aggressive"


class ExperienceLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class AssetClass(str, Enum):
    stock = "stock"
    etf = "etf"
    mutual_fund = "mutual_fund"
    bond = "bond"
    cash = "cash"
    real_estate = "real_estate"
    crypto = "crypto"
    other = "other"


class AgentName(str, Enum):
    finance_qa = "finance_qa"
    portfolio = "portfolio"
    market = "market"
    goal_planning = "goal_planning"
    news = "news"
    tax = "tax"


class UserProfile(BaseModel):
    user_id: str = Field(min_length=3, max_length=120, pattern=r"^[A-Za-z0-9._@\- ]+$")
    name: str | None = None
    risk_tolerance: RiskTolerance = RiskTolerance.moderate
    experience_level: ExperienceLevel = ExperienceLevel.beginner
    horizon_years: int = Field(default=10, ge=1, le=60)
    email: str | None = Field(default=None, max_length=254, description="Optional; enables self-service password reset")

    @field_validator("email")
    @classmethod
    def _valid_email(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        local, _, domain = value.partition("@")
        if not local or "." not in domain or domain.startswith(".") or domain.endswith(".") or " " in value:
            raise ValueError("Enter a valid email address.")
        return value.lower()


class Holding(BaseModel):
    ticker: str = Field(min_length=1, max_length=12)
    shares: float = Field(ge=0)
    cost_basis: float | None = Field(default=None, ge=0, description="Total cost, USD")
    asset_class: AssetClass = AssetClass.stock
    expense_ratio: float | None = Field(default=None, ge=0, le=0.1, description="Annual, as a fraction")

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class Portfolio(BaseModel):
    user_id: str
    holdings: list[Holding] = Field(default_factory=list)
    cash: float = Field(default=0.0, ge=0)


class PortfolioMetrics(BaseModel):
    total_value: float
    allocation_by_asset_class: dict[str, float] = Field(default_factory=dict)
    allocation_by_holding: dict[str, float] = Field(default_factory=dict)
    invested_value: float = 0.0
    cash_value: float = 0.0
    total_cost_basis: float | None = None
    unrealized_gain_loss: float | None = None
    unrealized_gain_pct: float | None = None
    weighted_expense_ratio: float | None = Field(default=None, description="Annual, as a percent (0.05 = 0.05%)")
    annual_fee_estimate: float | None = None
    effective_positions: float = 0.0
    diversification_score: float = Field(ge=0, le=100)
    risk_score: float = Field(default=0.0, ge=0, le=100)
    risk_label: str
    notes: list[str] = Field(default_factory=list)
    as_of: datetime = Field(default_factory=_utcnow)


class Goal(BaseModel):
    name: str = "My goal"
    target_amount: float = Field(gt=0)
    target_date: date
    current_savings: float = Field(default=0.0, ge=0)
    monthly_contribution: float = Field(default=0.0, ge=0)
    risk_tolerance: RiskTolerance = RiskTolerance.moderate
    assumed_return_pct: float = Field(default=0.0, ge=-50, le=100,
                                      description="Your own assumed annual return (%); projected as a below/expected/above trio "
                                                  "(+/- a fixed spread). 0 by default (no assumed growth).")


class GoalScenario(BaseModel):
    label: str
    annual_return: float
    projected_value: float
    real_projected_value: float | None = Field(default=None, description="In today's dollars (inflation-adjusted)")
    on_track: bool
    shortfall: float = 0.0
    required_monthly: float = Field(description="TOTAL monthly contribution needed to reach the target (not extra on top of current)")
    years_to_goal: float | None = Field(default=None, description="At current contributions; None if not reached in 100 years")


class GoalProjection(BaseModel):
    goal: Goal
    years: float
    scenarios: list[GoalScenario]
    yearly_path: dict[str, list[float]] = Field(default_factory=dict)


class Quote(BaseModel):
    symbol: str
    price: float
    change_pct: float | None = None
    previous_close: float | None = None
    currency: str = "USD"
    as_of: datetime = Field(default_factory=_utcnow, description="When this value was fetched/produced")
    trading_day: date | None = Field(default=None, description="Market date the price belongs to, if known")
    source: str = Field(description="alphavantage | yfinance | mock")
    stale: bool = Field(default=False, description="True when served from an old cache or the mock provider")
    cache_hit: bool = False
    data_status: DataStatus = DataStatus.live
    notice: str | None = Field(default=None, description="Plain-language disclosure; always set unless data_status is live")

    @model_validator(mode="after")
    def _always_disclose(self) -> "Quote":
        """Sample data and stale flags can never travel without a user-facing notice."""
        if self.source in ("mock", "sample") and self.data_status is not DataStatus.sample:
            self.data_status = DataStatus.sample
        if self.stale and self.data_status is DataStatus.live:
            self.data_status = DataStatus.stale
        if self.data_status is not DataStatus.live:
            self.stale = self.stale or self.data_status is not DataStatus.cached
            if not self.notice:
                self.notice = build_notice(self.data_status, self.source, self.age_seconds())
        return self

    def age_seconds(self) -> float:
        now = datetime.now(timezone.utc)
        as_of = self.as_of if self.as_of.tzinfo else self.as_of.replace(tzinfo=timezone.utc)
        return max(0.0, (now - as_of).total_seconds())

    def with_status(self, status: DataStatus, age_seconds: float | None = None) -> "Quote":
        """Copy of this quote re-labelled (e.g. served from cache) with an accurate notice.

        `age_seconds` should be the cache entry's age; it defaults to the time since `as_of`.
        """
        age = self.age_seconds() if age_seconds is None else age_seconds
        update = {
            "data_status": status,
            "cache_hit": status in (DataStatus.cached, DataStatus.stale),
            "stale": status in (DataStatus.stale, DataStatus.sample),
            "notice": build_notice(status, self.source, age),
        }
        return self.model_copy(update=update)


class PricePoint(BaseModel):
    day: date
    close: float = Field(ge=0)


class TrendSummary(BaseModel):
    symbol: str
    points: int
    last_close: float
    sma_20: float | None = None
    sma_50: float | None = None
    return_1m_pct: float | None = None
    return_3m_pct: float | None = None
    volatility_annual_pct: float | None = None
    max_drawdown_pct: float | None = None
    period_high: float
    period_low: float
    trend_label: str
    notes: list[str] = Field(default_factory=list)
    data_status: DataStatus = DataStatus.live
    notice: str | None = Field(default=None, description="Set whenever the underlying history is not live")


class NewsItem(BaseModel):
    title: str
    url: str
    source: str | None = None
    published: str | None = None
    snippet: str | None = None
    provider: str = ""


class Citation(BaseModel):
    index: int
    title: str
    source: str | None = None
    url: str | None = None
    category: str | None = None
    excerpt: str | None = None
    score: float | None = None


class AgentResult(BaseModel):
    agent: AgentName
    content: str = ""
    citations: list[Citation] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    error_code: str | None = Field(default=None, description="Machine-readable code, e.g. 'llm_unavailable'")
    cache_hit: str | None = Field(default=None, description="'exact', 'semantic', or None")
    data_status: DataStatus = DataStatus.live
    notice: str | None = Field(default=None, description="Plain-language disclosure when data is cached/stale/sample")

    @property
    def ok(self) -> bool:
        return self.error is None
