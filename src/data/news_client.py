"""Financial news: SerpAPI Google News (primary) -> yFinance news -> last-good cache -> sample headlines.

Privacy: the search query may be derived from user text, so it is passed through the PII redactor
before leaving the machine. Free-tier SerpAPI is ~100 searches/month, tracked persistently.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import Any, Callable, Protocol

import httpx

from src.core.config import get_config, settings
from src.core.errors import ProviderError, ProviderUnavailableError
from src.core.freshness import DataStatus, build_notice
from src.core.models import NewsItem
from src.data.cache import PersistentCache, UsageTracker
from src.data.db import Database
from src.utils.rate_limit import QuotaGuard
from src.utils.retry import RetryableError, retry_call

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 200
DEFAULT_LIMIT = 8


class NewsProvider(Protocol):
    name: str
    is_sample: bool

    def search(self, query: str, limit: int) -> list[NewsItem]: ...


_STOPWORDS = frozenset("""a an and are as at be between but by can could did do does for from how i if in is it its me my of on or
should so than that the their them then there these they this to us vs versus was we what when where which who why will with would you
your about explain tell please difference compare best good better worth""".split())


def keyword_query(query: str, max_words: int = 6) -> str:
    """Reduce a natural-language question to search keywords (Google News prefers keywords)."""
    words = re.findall(r"[A-Za-z0-9$.&'-]+", query)
    kept = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 1]
    return " ".join(kept[:max_words])


def _clean_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()[:MAX_QUERY_CHARS]


_PUBLISHED_FORMATS = ("%m/%d/%Y, %I:%M %p, %z %Z", "%m/%d/%Y, %I:%M %p", "%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z")


def _parse_published(value: str | None) -> datetime | None:
    """Best-effort parse of the `published` free-text field (format varies by provider); None if unparseable."""
    text = (value or "").strip()
    if not text:
        return None
    if text.isdigit():  # yFinance providerPublishTime: epoch seconds (or milliseconds)
        try:
            ts = int(text)
            return datetime.fromtimestamp(ts / 1000 if ts > 10_000_000_000 else ts, tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            return None
    try:
        dt = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in _PUBLISHED_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = parsedate_to_datetime(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _sort_by_published_desc(items: list[NewsItem]) -> list[NewsItem]:
    """Latest first. Items whose date can't be parsed sort last, in their original (provider) order."""
    return sorted(items, key=lambda it: _parse_published(it.published) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)


class SerpApiNewsProvider:
    name = "serpapi"
    is_sample = False
    BASE_URL = "https://serpapi.com/search.json"

    def __init__(self, api_key: str, guard: QuotaGuard, *, timeout: float = 15, attempts: int = 3,
                 base_delay: float = 0.5, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._key, self.guard = api_key, guard
        self._timeout, self._attempts, self._base_delay = timeout, attempts, base_delay
        self._client, self._sleep = client, sleep

    def search(self, query: str, limit: int) -> list[NewsItem]:
        if not self._key:
            raise ProviderUnavailableError("SerpAPI key is not configured")

        def attempt() -> httpx.Response:
            if not self.guard.try_acquire():
                raise ProviderUnavailableError("SerpAPI monthly quota reached")
            try:
                client = self._client or httpx
                response = client.get(
                    self.BASE_URL,
                    params={"engine": "google_news", "q": query, "gl": "us", "hl": "en", "api_key": self._key},
                    timeout=self._timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise RetryableError(type(exc).__name__) from None
            if response.status_code >= 500 or response.status_code == 429:
                raise RetryableError(f"HTTP {response.status_code}")
            return response

        try:
            response = retry_call(attempt, attempts=self._attempts, base_delay=self._base_delay, sleep=self._sleep)
        except RetryableError as exc:
            raise ProviderUnavailableError(f"SerpAPI request failed ({exc})") from None
        if response.status_code != 200:
            if response.status_code in (401, 403):
                raise ProviderUnavailableError("SerpAPI rejected the API key")
            raise ProviderUnavailableError(f"SerpAPI returned HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise ProviderUnavailableError("SerpAPI returned invalid JSON") from None
        if not isinstance(data, dict) or data.get("error"):
            error = str(data.get("error", "")) if isinstance(data, dict) else ""
            if "hasn't returned any results" in error or "no results" in error.lower():
                return []
            raise ProviderUnavailableError("SerpAPI reported an error")
        items = []
        for row in data.get("news_results") or []:
            # Stories may be nested under `stories`; take the top-level item if it has a link.
            candidate = row if row.get("link") else (row.get("stories") or [{}])[0]
            title, link = candidate.get("title"), candidate.get("link")
            if not title or not link:
                continue
            source = candidate.get("source")
            items.append(NewsItem(
                title=str(title)[:300], url=str(link),
                source=(source.get("name") if isinstance(source, dict) else source),
                published=candidate.get("date"), snippet=(candidate.get("snippet") or None), provider=self.name))
            if len(items) >= limit:
                break
        return items


# Common all-caps finance terms that are not tickers (avoid looking up "ETF" or "IRA" as symbols).
_NOT_TICKERS = frozenset({"ETF", "ETFS", "IRA", "SEC", "GDP", "CPI", "CEO", "CFO", "FED", "USA", "API", "LLC", "IPO",
                          "ESG", "REIT", "ROTH", "HSA", "AI", "US", "UK", "EU", "FAQ", "PDF", "USD", "APR", "APY", "NAV"})
_TICKER_TOKEN = re.compile(r"^\$?([A-Z]{2,5}(?:[.\-][A-Z]{1,4})?)[,.?!:;]?$")


def _find_ticker(query: str) -> str | None:
    """First explicit ticker in the query: an ALL-CAPS token (2-5 letters) or a $-prefixed one.

    Lowercase words never count, so 'what is an etf' or 'index funds' do not become symbols.
    """
    for token in query.split():
        dollar = token.startswith("$")
        match = _TICKER_TOKEN.match(token) or (re.match(r"^\$([A-Za-z]{1,5}(?:[.\-][A-Za-z]{1,4})?)[,.?!:;]?$", token) if dollar else None)
        if match:
            symbol = match.group(1).upper()
            if dollar or symbol not in _NOT_TICKERS:
                return symbol
    return None


class YFinanceNewsProvider:
    """Ticker-based news from Yahoo (handles both the older flat and newer `content` item formats)."""

    name = "yfinance"
    is_sample = False

    def __init__(self, ticker_factory: Callable[[str], Any] | None = None) -> None:
        self._factory = ticker_factory

    def search(self, query: str, limit: int) -> list[NewsItem]:
        symbol = _find_ticker(query)
        if symbol is None:
            raise ProviderUnavailableError("yFinance news needs a ticker symbol")
        try:
            if self._factory is None:
                import yfinance as yf

                self._factory = yf.Ticker
            raw = self._factory(symbol).news or []
        except Exception as exc:
            raise ProviderUnavailableError(f"yFinance news failed ({type(exc).__name__})") from None
        items = []
        for entry in raw:
            content = entry.get("content", entry) if isinstance(entry, dict) else {}
            title = content.get("title")
            url = (content.get("canonicalUrl") or {}).get("url") or content.get("link") or (content.get("clickThroughUrl") or {}).get("url")
            if not title or not url:
                continue
            publisher = content.get("publisher") or (content.get("provider") or {}).get("displayName")
            items.append(NewsItem(title=str(title)[:300], url=str(url), source=publisher,
                                  published=str(content.get("pubDate") or content.get("providerPublishTime") or "") or None,
                                  snippet=content.get("summary") or None, provider=self.name))
            if len(items) >= limit:
                break
        return items


class SampleNewsProvider:
    """Static, clearly-labelled sample headlines for when live providers are down."""

    name = "sample"
    is_sample = True

    def search(self, query: str, limit: int) -> list[NewsItem]:
        base = [
            ("Sample: How diversification can reduce single-company risk", "https://www.investor.gov/introduction-investing/investing-basics/glossary/diversification"),
            ("Sample: Understanding index funds and expense ratios", "https://www.bogleheads.org/wiki/Index_fund"),
            ("Sample: Why inflation matters for long-term savings", "https://www.investor.gov/introduction-investing/investing-basics"),
        ]
        return [NewsItem(title=t, url=u, source="Sample content", snippet="Live news is unavailable right now.",
                         provider=self.name) for t, u in base[:limit]]


class NewsService:
    def __init__(self, providers: list[NewsProvider], cache: PersistentCache, *, ttl_seconds: float,
                 stale_max_age_seconds: float, redactor: Callable[[str], str] | None = None) -> None:
        self.providers, self.cache = providers, cache
        self.ttl, self.stale_max_age = ttl_seconds, stale_max_age_seconds
        self._redactor = redactor

    def _redact(self, text: str) -> str:
        if self._redactor is not None:
            return self._redactor(text)
        from src.core.pii import redact_text

        return redact_text(text).text

    def search(self, query: str, limit: int = DEFAULT_LIMIT) -> tuple[list[NewsItem], dict[str, Any]]:
        """Returns (items, meta) where meta = {source, stale, cache_hit, query}. Never sends raw PII out."""
        cleaned = _clean_query(query or "")
        if not cleaned:
            cleaned = "stock market"
        safe_query = _clean_query(self._redact(cleaned))  # fail closed: redactor errors propagate
        key = "news:" + hashlib.sha256(f"{safe_query.lower()}|{limit}".encode()).hexdigest()[:24]

        def pack(items: list[NewsItem]) -> dict[str, Any]:
            return {"items": [i.model_dump(mode="json") for i in items]}

        empty_key = key + ":empty"  # separate key so a "no results" note never overwrites a last-good value

        def meta(source: str | None, status: DataStatus, *, cache_hit: bool, age: float | None = None) -> dict[str, Any]:
            return {"source": source, "stale": status in (DataStatus.stale, DataStatus.sample), "cache_hit": cache_hit,
                    "query": safe_query, "data_status": status.value, "notice": build_notice(status, source, age, kind="news")}

        fresh = self.cache.get_with_age(key, self.ttl)
        if fresh is not None:
            items = _sort_by_published_desc([NewsItem.model_validate(i) for i in fresh[0]["items"]])
            return items, meta(items[0].provider if items else "cache", DataStatus.cached, cache_hit=True, age=fresh[1])

        # Providers recently returned nothing for this query: do not spend more quota on it.
        if self.cache.get(empty_key, self.ttl) is not None:
            return self._fallback(key, safe_query, limit, meta, cache_hit=True)

        keywords = keyword_query(safe_query)
        answered_empty = False
        for provider in (p for p in self.providers if not p.is_sample):
            try:
                items = provider.search(safe_query, limit)
                if not items and keywords and keywords.lower() != safe_query.lower():
                    items = provider.search(keywords, limit)  # natural-language questions often return nothing
            except ProviderError as exc:
                logger.warning("%s news unavailable: %s", provider.name, exc)
                continue
            except Exception:
                logger.exception("Unexpected %s news failure", provider.name)
                continue
            if items:
                items = _sort_by_published_desc(items)
                self.cache.set(key, pack(items))
                return items, meta(provider.name, DataStatus.live, cache_hit=False)
            answered_empty = True
        if answered_empty:
            self.cache.set(empty_key, True)
        return self._fallback(key, safe_query, limit, meta, cache_hit=False)

    def _fallback(self, key: str, safe_query: str, limit: int, meta, *, cache_hit: bool) -> tuple[list[NewsItem], dict[str, Any]]:
        """Last-good cached items (any age within the stale window), else clearly-labelled sample headlines."""
        stale = self.cache.get_stale(key, self.stale_max_age)
        if stale is not None and stale[0]["items"]:
            items = _sort_by_published_desc([NewsItem.model_validate(i) for i in stale[0]["items"]])
            return items, meta(items[0].provider, DataStatus.stale, cache_hit=True, age=stale[1])
        for provider in (p for p in self.providers if p.is_sample):
            return provider.search(safe_query, limit), meta(provider.name, DataStatus.sample, cache_hit=cache_hit)
        return [], meta(None, DataStatus.sample, cache_hit=cache_hit)


def build_news_service(db: Database | None = None) -> NewsService:
    cfg = get_config()
    db = db or Database(settings.app_db_path)
    db.init_schema()
    usage = UsageTracker(db)
    providers: list[NewsProvider] = []
    if settings.serpapi_api_key:
        guard = QuotaGuard("serpapi", period_limit=cfg.market.serpapi_monthly_limit, period_kind="month", usage=usage)
        providers.append(SerpApiNewsProvider(settings.serpapi_api_key, guard, attempts=cfg.market.max_retries,
                                             base_delay=cfg.market.retry_base_delay_seconds))
    else:
        logger.warning("SERPAPI_API_KEY not set; skipping SerpAPI news")
    providers += [YFinanceNewsProvider(), SampleNewsProvider()]
    return NewsService(providers, PersistentCache(db), ttl_seconds=cfg.cache.news_ttl_seconds,
                       stale_max_age_seconds=cfg.market.stale_max_age_seconds)


@lru_cache(maxsize=1)
def get_news_service() -> NewsService:
    return build_news_service()
