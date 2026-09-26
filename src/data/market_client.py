"""Market data: Alpha Vantage (primary) -> yFinance (fallback) -> last-good cache -> sample data.

Chain semantics for a quote:
  1. fresh cache hit (TTL 30 min)                       -> return
  2. each real provider in configured order             -> first success is cached and returned
       * SymbolNotFound      -> try the next provider (coverage differs, e.g. crypto is yFinance-only)
       * ProviderUnavailable -> try the next provider (network/5xx/quota/rate limit)
  3. every provider that answered said "not found" and none were down -> SymbolNotFoundError
  4. last-good cached value (any age up to stale_max_age)  -> returned flagged stale
  5. sample (mock) provider, flagged stale + source "mock" -> only when providers are DOWN
Sample data is never cached, so it can never be mistaken for a real last-good value.

API keys are sent only to their own provider and are scrubbed from every error message.
"""
from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Callable, Protocol

import httpx

from src.core.config import MarketConfig, get_config, settings
from src.core.errors import ProviderError, ProviderUnavailableError, SymbolNotFoundError
from src.core.freshness import DataStatus, build_notice
from src.core.market_calc import compute_trend
from src.core.models import PricePoint, Quote, TrendSummary
from src.data.cache import PersistentCache, UsageTracker
from src.data.db import Database
from src.utils.rate_limit import QuotaGuard
from src.utils.retry import RetryableError, retry_call
from src.utils.validators import normalize_ticker

logger = logging.getLogger(__name__)

QUOTE_TTL_KEY = "market_ttl_seconds"
MAX_BATCH = 25


class MarketProvider(Protocol):
    name: str
    is_sample: bool

    def supports(self, symbol: str) -> bool: ...
    def get_quote(self, symbol: str) -> Quote: ...
    def get_history(self, symbol: str, days: int) -> list[PricePoint]: ...


def _is_crypto_pair(symbol: str) -> bool:
    return symbol.endswith(("-USD", "-EUR", "-GBP", "-BTC", "-ETH"))


# ------------------------------------------------------------------------------ Alpha Vantage
class AlphaVantageProvider:
    """Alpha Vantage free tier: ~25 requests/day, 5/minute, US equities/ETFs (no crypto pairs)."""

    name = "alphavantage"
    is_sample = False
    BASE_URL = "https://www.alphavantage.co/query"

    MIN_INTERVAL_SECONDS = 1.1  # free tier allows 1 request/second

    def __init__(
        self,
        api_key: str,
        guard: QuotaGuard,
        *,
        timeout: float = 10,
        attempts: int = 3,
        base_delay: float = 0.5,
        min_interval: float = MIN_INTERVAL_SECONDS,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._key = api_key
        self.guard = guard
        self._timeout = timeout
        self._attempts = attempts
        self._base_delay = base_delay
        self._min_interval = min_interval
        self._client = client
        self._sleep = sleep
        self._clock = clock
        self._pace_lock = threading.Lock()
        self._last_call: float | None = None

    def supports(self, symbol: str) -> bool:
        return not _is_crypto_pair(symbol)

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def _pace(self) -> None:
        """Keep at least `min_interval` seconds between requests to stay under the burst limit."""
        with self._pace_lock:
            if self._last_call is not None and self._min_interval > 0:
                wait = self._min_interval - (self._clock() - self._last_call)
                if wait > 0:
                    self._sleep(wait)
            self._last_call = self._clock()

    def _call(self, params: dict[str, str]) -> dict[str, Any]:
        if not self._key:
            raise ProviderUnavailableError("Alpha Vantage API key is not configured")

        def attempt() -> dict[str, Any]:
            if not self.guard.try_acquire():  # not retryable: fail fast so the chain can fall back
                raise ProviderUnavailableError("Alpha Vantage quota or rate limit reached")
            self._pace()
            try:
                client = self._client or httpx
                response = client.get(self.BASE_URL, params={**params, "apikey": self._key}, timeout=self._timeout)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise RetryableError(self._scrub(f"{type(exc).__name__}")) from None
            if response.status_code >= 500 or response.status_code == 429:
                raise RetryableError(f"HTTP {response.status_code}")
            if response.status_code != 200:
                raise ProviderUnavailableError(f"Alpha Vantage returned HTTP {response.status_code}")
            try:
                data = response.json()
            except ValueError:
                raise ProviderUnavailableError("Alpha Vantage returned invalid JSON") from None
            if not isinstance(data, dict):
                raise ProviderUnavailableError("Alpha Vantage returned an unexpected payload")

            message = data.get("Note") or data.get("Information")
            if message:
                lowered = str(message).lower()
                # The burst message ALSO mentions the daily limit, so check for it first.
                if "spreading out" in lowered or "per second" in lowered or "calls per minute" in lowered:
                    raise RetryableError("Alpha Vantage burst limit")
                if "per day" in lowered or "daily" in lowered:
                    self.guard.mark_exhausted()
                raise ProviderUnavailableError("Alpha Vantage quota or key problem (see provider message)")
            if "Error Message" in data:
                raise SymbolNotFoundError("Alpha Vantage does not recognize that symbol")
            return data

        try:
            return retry_call(attempt, attempts=self._attempts, base_delay=self._base_delay, sleep=self._sleep)
        except RetryableError as exc:
            raise ProviderUnavailableError(f"Alpha Vantage request failed ({exc})") from None

    def get_quote(self, symbol: str) -> Quote:
        data = self._call({"function": "GLOBAL_QUOTE", "symbol": symbol})
        raw = data.get("Global Quote") or {}
        if "05. price" not in raw:
            raise SymbolNotFoundError("Alpha Vantage has no quote for that symbol")
        try:
            price = float(raw["05. price"])
            previous = float(raw["08. previous close"]) if raw.get("08. previous close") else None
            change = raw.get("10. change percent", "").strip().rstrip("%")
            change_pct = round(float(change), 4) if change else None
            day = date.fromisoformat(raw["07. latest trading day"]) if raw.get("07. latest trading day") else None
        except (ValueError, KeyError):
            raise ProviderUnavailableError("Alpha Vantage returned a malformed quote") from None
        return Quote(symbol=symbol, price=price, change_pct=change_pct, previous_close=previous,
                     trading_day=day, source=self.name)

    def get_history(self, symbol: str, days: int) -> list[PricePoint]:
        data = self._call({"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "compact"})
        series = data.get("Time Series (Daily)")
        if not series:
            raise SymbolNotFoundError("Alpha Vantage has no history for that symbol")
        cutoff = date.today() - timedelta(days=days)
        points = []
        try:
            for day_str, row in series.items():
                d = date.fromisoformat(day_str)
                if d >= cutoff:
                    points.append(PricePoint(day=d, close=float(row["4. close"])))
        except (ValueError, KeyError):
            raise ProviderUnavailableError("Alpha Vantage returned malformed history") from None
        return sorted(points, key=lambda p: p.day)


# ---------------------------------------------------------------------------------- yFinance
class YFinanceProvider:
    """yFinance (unofficial Yahoo endpoint): no key, covers crypto pairs; can break without notice."""

    name = "yfinance"
    is_sample = False

    def __init__(
        self,
        *,
        ticker_factory: Callable[[str], Any] | None = None,
        attempts: int = 2,
        base_delay: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._factory = ticker_factory
        self._attempts = attempts
        self._base_delay = base_delay
        self._sleep = sleep

    def supports(self, symbol: str) -> bool:
        return True

    def _ticker(self, symbol: str):
        if self._factory is None:
            import yfinance as yf

            self._factory = yf.Ticker
        return self._factory(symbol)

    def _run(self, fn: Callable[[], Any]) -> Any:
        def attempt():
            try:
                return fn()
            except SymbolNotFoundError:
                raise
            except Exception as exc:  # yfinance/curl_cffi raise many types for network problems
                raise RetryableError(type(exc).__name__) from None

        try:
            return retry_call(attempt, attempts=self._attempts, base_delay=self._base_delay, sleep=self._sleep)
        except RetryableError as exc:
            raise ProviderUnavailableError(f"yFinance request failed ({exc})") from None

    @staticmethod
    def _num(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(number) or math.isinf(number) else number

    def get_quote(self, symbol: str) -> Quote:
        def fetch():
            info = self._ticker(symbol).fast_info
            last = self._num(info["last_price"])
            if last is None or last <= 0:
                raise SymbolNotFoundError("yFinance has no price for that symbol")
            prev = self._num(info["previous_close"])
            return last, prev

        last, prev = self._run(fetch)
        change_pct = round((last / prev - 1) * 100, 4) if prev else None
        return Quote(symbol=symbol, price=round(last, 4), change_pct=change_pct,
                     previous_close=round(prev, 4) if prev else None, source=self.name)

    def get_history(self, symbol: str, days: int) -> list[PricePoint]:
        def fetch():
            frame = self._ticker(symbol).history(start=(date.today() - timedelta(days=days)).isoformat(), auto_adjust=True)
            if frame is None or len(frame) == 0:
                raise SymbolNotFoundError("yFinance has no history for that symbol")
            return frame

        frame = self._run(fetch)
        points = []
        for idx, close in frame["Close"].items():
            value = self._num(close)
            if value is not None and value >= 0:
                points.append(PricePoint(day=idx.date() if hasattr(idx, "date") else idx, close=round(value, 4)))
        if not points:
            raise SymbolNotFoundError("yFinance returned no usable history")
        return sorted(points, key=lambda p: p.day)


# ------------------------------------------------------------------------------ sample data
MOCK_BASE_PRICES: dict[str, float] = {
    "AAPL": 190.0, "MSFT": 420.0, "GOOGL": 170.0, "AMZN": 185.0, "META": 500.0, "NVDA": 120.0, "TSLA": 240.0,
    "NFLX": 640.0, "AMD": 160.0, "JPM": 200.0, "BAC": 40.0, "V": 275.0, "MA": 460.0, "JNJ": 155.0, "KO": 63.0,
    "PEP": 170.0, "WMT": 68.0, "DIS": 100.0, "XOM": 115.0, "COIN": 250.0, "O": 57.0,
    "VTI": 265.0, "VOO": 490.0, "SPY": 530.0, "QQQ": 450.0, "VXUS": 62.0, "BND": 73.0, "AGG": 98.0, "TLT": 92.0,
    "SCHD": 79.0, "VNQ": 88.0, "IWM": 205.0, "GLD": 215.0, "TIP": 107.0, "VGSH": 58.0, "BTC-USD": 65000.0,
    "ETH-USD": 3300.0,
}


class MockProvider:
    """Deterministic sample data used ONLY when real providers are down. Always flagged stale."""

    name = "mock"
    is_sample = True

    def supports(self, symbol: str) -> bool:
        return symbol in MOCK_BASE_PRICES

    def _base(self, symbol: str) -> float:
        if symbol not in MOCK_BASE_PRICES:
            raise SymbolNotFoundError("No sample data for that symbol")
        return MOCK_BASE_PRICES[symbol]

    def get_quote(self, symbol: str) -> Quote:
        base = self._base(symbol)
        rng = random.Random(f"{symbol}:{date.today().isoformat()}")
        change_pct = round(rng.uniform(-1.5, 1.5), 2)
        price = round(base * (1 + change_pct / 100), 2)
        return Quote(symbol=symbol, price=price, change_pct=change_pct, previous_close=round(base, 2),
                     trading_day=date.today(), source=self.name, stale=True)

    def get_history(self, symbol: str, days: int) -> list[PricePoint]:
        base = self._base(symbol)
        rng = random.Random(f"hist:{symbol}")
        business_days = [date.today() - timedelta(days=i) for i in range(days) if (date.today() - timedelta(days=i)).weekday() < 5]
        business_days.reverse()
        price, series = base, []
        for _ in business_days:
            series.append(price)
            price *= 1 + rng.gauss(0.0003, 0.011)
        scale = base / series[-1]  # end the series at the base price
        return [PricePoint(day=d, close=round(v * scale, 2)) for d, v in zip(business_days, series)]


# ------------------------------------------------------------------------------------ service
@dataclass
class QuoteBatch:
    quotes: dict[str, Quote] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def notices(self) -> list[str]:
        """Unique disclosures for every non-live quote in the batch (empty when all are live)."""
        return list(dict.fromkeys(q.notice for q in self.quotes.values() if q.notice))


@dataclass
class History:
    symbol: str
    points: list[PricePoint]
    source: str
    stale: bool = False
    status: DataStatus = DataStatus.live
    notice: str | None = None
    age_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.source == "mock":
            self.status, self.stale = DataStatus.sample, True
        if self.status is not DataStatus.live and not self.notice:
            self.notice = build_notice(self.status, self.source, self.age_seconds, kind="history")


class MarketService:
    def __init__(
        self,
        providers: list[MarketProvider],
        cache: PersistentCache,
        config: MarketConfig | None = None,
        *,
        quote_ttl_seconds: float = 1800,
    ) -> None:
        self.providers = providers
        self.cache = cache
        self.config = config or get_config().market
        self.quote_ttl = quote_ttl_seconds
        self._health: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # -- bookkeeping ---------------------------------------------------------------------------
    def _record(self, provider: str, ok: bool, error: str | None = None) -> None:
        with self._lock:
            entry = self._health.setdefault(provider, {})
            entry["ok"] = ok
            entry["last_attempt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            entry["last_error"] = None if ok else error

    def status(self) -> list[dict[str, Any]]:
        out = []
        for p in self.providers:
            guard = getattr(p, "guard", None)
            out.append({"provider": p.name, "sample": p.is_sample,
                        "quota_remaining": guard.remaining() if guard else None, **self._health.get(p.name, {})})
        return out

    # -- generic chain -------------------------------------------------------------------------
    def _resolve(self, symbol: str, call: Callable[[MarketProvider], Any]) -> tuple[Any, MarketProvider | None, bool]:
        """Run `call` down the provider chain. Returns (value, provider, from_stale_needed).

        Raises SymbolNotFoundError when every answering provider says the symbol does not exist.
        Returns (None, None, True) when providers are down and the caller should try stale/sample data.
        """
        not_found = unavailable = 0
        for provider in (p for p in self.providers if not p.is_sample):
            if not provider.supports(symbol):
                continue
            try:
                value = call(provider)
            except SymbolNotFoundError:
                not_found += 1
                self._record(provider.name, True)  # provider is healthy, it just lacks the symbol
                continue
            except ProviderError as exc:
                unavailable += 1
                self._record(provider.name, False, str(exc))
                logger.warning("%s unavailable: %s", provider.name, exc)
                continue
            except Exception as exc:  # a provider bug must never take the app down
                unavailable += 1
                self._record(provider.name, False, type(exc).__name__)
                logger.exception("Unexpected %s failure", provider.name)
                continue
            self._record(provider.name, True)
            return value, provider, False
        if not_found and not unavailable:
            raise SymbolNotFoundError(f"Ticker {symbol} was not found")
        return None, None, True

    def _sample(self, symbol: str, call: Callable[[MarketProvider], Any]) -> Any:
        for provider in (p for p in self.providers if p.is_sample):
            try:
                return call(provider)
            except ProviderError:
                continue
        raise ProviderUnavailableError(f"No market data is available for {symbol} right now")

    # -- quotes --------------------------------------------------------------------------------
    def get_quote(self, symbol: str) -> Quote:
        sym = normalize_ticker(symbol)
        if sym is None:
            raise SymbolNotFoundError(f"'{str(symbol)[:20]}' is not a valid ticker symbol")
        key = f"quote:{sym}"
        fresh = self.cache.get_with_age(key, self.quote_ttl)
        if fresh is not None:
            return Quote.model_validate(fresh[0]).with_status(DataStatus.cached, fresh[1])

        quote, _provider, need_fallback = self._resolve(sym, lambda p: p.get_quote(sym))
        if quote is not None:
            self.cache.set(key, quote.model_dump(mode="json"))
            return quote

        stale = self.cache.get_stale(key, self.config.stale_max_age_seconds)
        if stale is not None:
            return Quote.model_validate(stale[0]).with_status(DataStatus.stale, stale[1])
        return self._sample(sym, lambda p: p.get_quote(sym))

    def get_quotes(self, symbols: list[str]) -> QuoteBatch:
        batch = QuoteBatch()
        seen: list[str] = []
        for raw in symbols[:MAX_BATCH]:
            sym = normalize_ticker(raw)
            if sym is None:
                batch.errors[str(raw)[:20]] = "Not a valid ticker symbol"
            elif sym not in seen:
                seen.append(sym)
        for sym in seen:
            try:
                batch.quotes[sym] = self.get_quote(sym)
            except ProviderError as exc:
                batch.errors[sym] = str(exc)
        if len(symbols) > MAX_BATCH:
            batch.errors["_limit"] = f"Only the first {MAX_BATCH} symbols were processed"
        return batch

    # -- history & trend -----------------------------------------------------------------------
    def get_history(self, symbol: str, days: int | None = None) -> History:
        sym = normalize_ticker(symbol)
        if sym is None:
            raise SymbolNotFoundError(f"'{str(symbol)[:20]}' is not a valid ticker symbol")
        days = days or self.config.history_days
        key = f"history:{sym}:{days}"

        def load(raw: dict[str, Any], status: DataStatus, age: float) -> History:
            return History(sym, [PricePoint.model_validate(p) for p in raw["points"]], raw["source"],
                           stale=status is DataStatus.stale, status=status, age_seconds=age)

        fresh = self.cache.get_with_age(key, self.config.history_ttl_seconds)
        if fresh is not None:
            return load(fresh[0], DataStatus.cached, fresh[1])
        result, provider, _ = self._resolve(sym, lambda p: p.get_history(sym, days))
        if result is not None:
            self.cache.set(key, {"source": provider.name, "points": [p.model_dump(mode="json") for p in result]})
            return History(sym, result, provider.name)
        stale = self.cache.get_stale(key, self.config.stale_max_age_seconds)
        if stale is not None:
            return load(stale[0], DataStatus.stale, stale[1])
        points = self._sample(sym, lambda p: p.get_history(sym, days))
        return History(sym, points, "mock", stale=True)

    def get_trend(self, symbol: str, days: int | None = None) -> tuple[TrendSummary, History]:
        history = self.get_history(symbol, days)
        trend = compute_trend(history.symbol, history.points)
        trend.data_status, trend.notice = history.status, history.notice
        if history.status is not DataStatus.live:
            trend.notes.insert(0, history.notice or "This trend is based on non-live data.")
        return trend, history


# ------------------------------------------------------------------------------------ factory
def build_market_service(db: Database | None = None, *, config: MarketConfig | None = None) -> MarketService:
    """Assemble the provider chain from config.yaml (order = priority) and the app database."""
    cfg = config or get_config().market
    app_config = get_config()
    db = db or Database(settings.app_db_path)
    db.init_schema()
    usage = UsageTracker(db)
    providers: list[MarketProvider] = []
    for name in cfg.providers:
        if name == "alphavantage":
            if not settings.alphavantage_api_key:
                logger.warning("ALPHAVANTAGE_API_KEY not set; skipping Alpha Vantage")
                continue
            guard = QuotaGuard("alphavantage", per_minute=cfg.alphavantage_calls_per_minute,
                               period_limit=cfg.alphavantage_daily_limit, period_kind="day", usage=usage)
            providers.append(AlphaVantageProvider(settings.alphavantage_api_key, guard,
                                                  timeout=cfg.request_timeout_seconds, attempts=cfg.max_retries,
                                                  base_delay=cfg.retry_base_delay_seconds))
        elif name == "yfinance":
            providers.append(YFinanceProvider(attempts=max(1, cfg.max_retries - 1), base_delay=cfg.retry_base_delay_seconds))
        elif name == "mock":
            providers.append(MockProvider())
        else:
            logger.warning("Unknown market provider '%s' ignored", name)
    return MarketService(providers, PersistentCache(db), cfg, quote_ttl_seconds=app_config.cache.market_ttl_seconds)


@lru_cache(maxsize=1)
def get_market_service() -> MarketService:
    return build_market_service()
