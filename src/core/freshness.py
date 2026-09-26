"""User-facing disclosure of where data came from.

Rule: whenever a value is NOT a fresh live lookup (served from cache, last-known copy, or sample
data) the user must be told, in plain language. Every result model that can be non-live builds its
`notice` from these helpers, so callers cannot forget: see Quote/TrendSummary validators and the
News/History services.
"""
from __future__ import annotations

from enum import Enum


class DataStatus(str, Enum):
    live = "live"       # fetched from the provider just now
    cached = "cached"   # served from a recent cache entry (still within its freshness window)
    stale = "stale"     # providers unavailable: last saved real value, older than the freshness window
    sample = "sample"   # providers unavailable and nothing saved: synthetic sample data, NOT real


PROVIDER_LABELS = {
    "alphavantage": "Alpha Vantage",
    "yfinance": "Yahoo Finance (via yFinance)",
    "serpapi": "Google News (via SerpAPI)",
    "mock": "sample data",
    "sample": "sample data",
}


def provider_label(source: str | None) -> str:
    if not source:
        return "an earlier lookup"
    return PROVIDER_LABELS.get(source, source)


def humanize_age(seconds: float | None) -> str:
    """'moments', '12 minutes', '3 hours', '2 days'."""
    if seconds is None or seconds < 0:
        return "an unknown amount of time"
    if seconds < 45:
        return "moments"
    minutes = round(seconds / 60)
    if minutes < 90:
        return f"{max(1, minutes)} minute{'s' if minutes != 1 else ''}"
    hours = round(seconds / 3600)
    if hours < 36:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = round(seconds / 86400)
    return f"{days} day{'s' if days != 1 else ''}"


def build_notice(status: DataStatus | str, source: str | None, age_seconds: float | None = None, *, kind: str = "price") -> str | None:
    """Plain-language notice for a non-live result; None when the data is live.

    kind: "price" (quotes), "history" (charts/trends), or "news".
    """
    status = DataStatus(status)
    if status is DataStatus.live:
        return None
    what = {"price": "data", "history": "price history", "news": "headlines"}.get(kind, "data")
    if status is DataStatus.sample:
        if kind == "news":
            return "Live news is unavailable right now. These are SAMPLE headlines for illustration, not current news."
        return f"Live {what} is unavailable right now. This is SAMPLE data for illustration only, not a real market figure."
    label, age = provider_label(source), humanize_age(age_seconds)
    how_old = "only moments old" if age == "moments" else f"about {age} old"
    if status is DataStatus.cached:
        return f"Cached {what} from {label}, {how_old}. This is not a fresh lookup."
    return f"Live {what} is unavailable right now. Showing the last saved {what} from {label}, {how_old}."


def build_qa_cache_notice(kind: str, age_seconds: float | None, matched_question: str | None = None) -> str:
    """Disclosure for a Finance Q&A answer served from the answer cache instead of being generated now.

    kind: "exact" (same question) or "semantic" (a similar earlier question).
    """
    age = humanize_age(age_seconds)
    how_old = "only moments old" if age == "moments" else f"about {age} old"
    if kind == "exact":
        return f"Answered from a saved response to this same question ({how_old}). It was not regenerated."
    quoted = ""
    if matched_question:
        shown = matched_question if len(matched_question) <= 120 else matched_question[:117].rstrip() + "..."
        quoted = f' ("{shown}")'
    return f"Answered from a saved response to a similar question{quoted} ({how_old}). It was not regenerated for your exact wording."


_SEVERITY = {DataStatus.live: 0, DataStatus.cached: 1, DataStatus.stale: 2, DataStatus.sample: 3}


def combine_status(statuses) -> DataStatus:
    """The 'worst' status among several results (sample > stale > cached > live), for one overall label."""
    worst = DataStatus.live
    for status in statuses:
        status = DataStatus(status)
        if _SEVERITY[status] > _SEVERITY[worst]:
            worst = status
    return worst
