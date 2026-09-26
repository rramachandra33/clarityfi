"""Request/response models for the REST API (kept separate from the chat schemas)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any

from fastapi import Path
from pydantic import BaseModel, Field, field_validator, model_validator

from src.core.freshness import DataStatus
from src.core.models import (ExperienceLevel, Goal, GoalProjection, Holding, NewsItem, PortfolioMetrics, PricePoint, Quote,
                             RiskTolerance, TrendSummary)

USER_ID_PATTERN = r"^[A-Za-z0-9._@\- ]+$"
UserIdPath = Annotated[str, Path(min_length=3, max_length=120, pattern=USER_ID_PATTERN, description="The user's id (3-120 characters).")]
MAX_HOLDINGS = 100
NOT_FOUND = {404: {"description": "Unknown user or item"}}
TOO_MANY = {429: {"description": "Rate limit exceeded (see the Retry-After header)"}}


# ------------------------------------------------------------------------------------------ users
class ProfileUpdate(BaseModel):
    risk_tolerance: RiskTolerance | None = None
    experience_level: ExperienceLevel | None = None
    horizon_years: int | None = Field(default=None, ge=1, le=60)


class EmailUpdate(BaseModel):
    email: str | None = Field(default=None, max_length=254, description="Set to null to remove; enables password reset")


# --------------------------------------------------------------------------------------- portfolio
class PortfolioIn(BaseModel):
    holdings: list[Holding] = Field(default_factory=list, max_length=MAX_HOLDINGS)
    cash: float = Field(default=0.0, ge=0, le=1e12)

    @model_validator(mode="after")
    def _unique_tickers(self) -> "PortfolioIn":
        seen = [h.ticker for h in self.holdings]
        dupes = sorted({t for t in seen if seen.count(t) > 1})
        if dupes:
            raise ValueError(f"Each ticker may appear once (combine the shares): {', '.join(dupes)}")
        return self


class PortfolioOut(BaseModel):
    user_id: str
    holdings: list[Holding]
    cash: float
    synthetic: bool = False


class PortfolioAnalysisOut(BaseModel):
    user_id: str
    empty: bool = False
    what_if: bool = Field(default=False, description="True when the analysis was of a submitted portfolio that was NOT saved")
    metrics: PortfolioMetrics | None = None
    holdings: list[dict[str, Any]] = Field(default_factory=list, description="Per-holding table, largest first")
    data_status: DataStatus = DataStatus.live
    notice: str | None = None
    notices: list[str] = Field(default_factory=list)
    priced_with: list[str] = Field(default_factory=list)
    unpriced: list[str] = Field(default_factory=list)
    sample_quotes_ignored: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)
    stated_risk_tolerance: str | None = None
    explanation: str | None = Field(default=None, description="Plain-language narration (only when explain=true)")


# ------------------------------------------------------------------------------------------ market
class QuotesOut(BaseModel):
    quotes: list[Quote]
    errors: dict[str, str] = Field(default_factory=dict)
    notices: list[str] = Field(default_factory=list, description="Disclosures for any cached / last-known / SAMPLE quote")
    data_status: DataStatus = DataStatus.live


class TrendOut(BaseModel):
    symbol: str
    trend: TrendSummary
    points: list[PricePoint] = Field(description="Closing prices, oldest first, for charting")
    source: str
    data_status: DataStatus = DataStatus.live
    notice: str | None = None


class ContextOut(BaseModel):
    symbol: str
    benchmark: str | None = None
    benchmark_label: str | None = None
    trend: TrendSummary
    benchmark_trend: TrendSummary | None = None
    relative: dict[str, float | None] = Field(default_factory=dict, description="Percentage-point gaps vs the benchmark")
    insights: list[str] = Field(description="Plain-language reading of the computed numbers (descriptive, not advice)")
    comparison: dict[str, list[dict[str, Any]]] = Field(default_factory=dict, description="Each series rebased to 100 on shared dates")
    data_status: DataStatus = DataStatus.live
    notices: list[str] = Field(default_factory=list)


class SegmentOut(BaseModel):
    symbol: str
    label: str
    description: str
    last_close: float | None = None
    return_1m_pct: float | None = None
    return_3m_pct: float | None = None
    volatility_annual_pct: float | None = None
    trend_label: str | None = None
    data_status: DataStatus = DataStatus.live
    source: str | None = None
    error: str | None = None


class OverviewOut(BaseModel):
    segments: list[SegmentOut]
    data_status: DataStatus = DataStatus.live
    notices: list[str] = Field(default_factory=list)
    takeaways: list[str] = Field(default_factory=list, description="Plain-language comparison across segments")


class PortfolioComparisonOut(BaseModel):
    user_id: str
    empty: bool = False
    benchmark: str | None = None
    included: list[str] = Field(default_factory=list, description="Holdings whose price history was used")
    excluded: dict[str, str] = Field(default_factory=dict, description="Holdings left out, with the reason")
    coverage_pct: float = 0.0
    days: int = 180
    portfolio_trend: TrendSummary | None = None
    benchmark_trend: TrendSummary | None = None
    relative: dict[str, float | None] = Field(default_factory=dict)
    comparison: dict[str, list[dict[str, Any]]] = Field(default_factory=dict, description="Your mix and the benchmark, each rebased to 100")
    insights: list[str] = Field(default_factory=list)
    data_status: DataStatus = DataStatus.live
    notices: list[str] = Field(default_factory=list)


class ProviderStatusOut(BaseModel):
    provider: str
    sample: bool
    quota_remaining: int | None = None
    ok: bool | None = None
    last_error: str | None = None
    last_attempt: str | None = None


# ------------------------------------------------------------------------------------------- goals
class GoalIn(BaseModel):
    name: str = Field(default="My goal", min_length=1, max_length=80)
    target_amount: float = Field(gt=0, le=1e10)
    target_date: date | None = None
    years: float | None = Field(default=None, gt=0, le=80, description="Alternative to target_date: years from today")
    current_savings: float = Field(default=0.0, ge=0, le=1e10)
    monthly_contribution: float = Field(default=0.0, ge=0, le=1e8)
    risk_tolerance: RiskTolerance | None = None
    assumed_return_pct: float = Field(default=0.0, ge=-50, le=100,
                                      description="Your own assumed annual return (%), projected as a below/expected/above trio "
                                                  "(+/- a fixed spread). 0 by default (no assumed growth).")

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        cleaned = "".join(ch for ch in v if ch.isprintable()).strip()
        if not cleaned:
            raise ValueError("name must not be blank")
        return cleaned

    @model_validator(mode="after")
    def _one_horizon(self) -> "GoalIn":
        if (self.target_date is None) == (self.years is None):
            raise ValueError("Provide exactly one of target_date or years")
        return self


class ProjectionIn(GoalIn):
    user_id: str | None = Field(default=None, min_length=3, max_length=120, pattern=USER_ID_PATTERN,
                                description="Optional: uses this user's risk tolerance when none is given")
    explain: bool = Field(default=False, description="Also return a plain-language explanation (uses the LLM)")


class GoalOut(BaseModel):
    id: int
    goal: Goal


class ProjectionOut(BaseModel):
    goal: Goal
    projection: GoalProjection
    assumptions: list[str]
    on_track_expected: bool
    explanation: str | None = None
    explanation_error: str | None = None


# -------------------------------------------------------------------------------------------- news
class NewsOut(BaseModel):
    query: str
    items: list[NewsItem]
    source: str | None = None
    data_status: DataStatus = DataStatus.live
    notice: str | None = None
    stale: bool = False
    cache_hit: bool = False
    summary: str | None = Field(default=None, description="Narrated summary (only when summarize=true)")


# ----------------------------------------------------------------------------------- conversations
class MessageOut(BaseModel):
    role: str
    content: str
    agent: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    cache_hit: str | None = None
    created_at: datetime | str | None = None


class ConversationOut(BaseModel):
    user_id: str
    conversation_id: str
    messages: list[MessageOut]
