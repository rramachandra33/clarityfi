"""Rule under test: whenever data is cached, last-known, or sample, the user is told."""
from datetime import datetime, timedelta, timezone

import pytest

from src.core.config import MarketConfig
from src.core.disclaimers import with_data_notices
from src.core.errors import ProviderUnavailableError
from src.core.freshness import DataStatus, build_notice, humanize_age, provider_label
from src.core.models import PricePoint, Quote
from src.data.cache import PersistentCache
from src.data.db import Database
from src.data.market_client import History, MarketService, MockProvider
from src.data.news_client import NewsService, SampleNewsProvider
from src.core.models import NewsItem


# ----------------------------------------------------------------------------- wording helpers
@pytest.mark.parametrize("seconds,text", [
    (0, "moments"), (44, "moments"), (60, "1 minute"), (600, "10 minutes"), (89 * 60, "89 minutes"),
    (91 * 60, "2 hours"), (3 * 3600, "3 hours"), (35 * 3600, "35 hours"), (3 * 86400, "3 days"),
    (None, "an unknown amount of time"), (-5, "an unknown amount of time"),
])
def test_humanize_age(seconds, text):
    assert humanize_age(seconds) == text


def test_provider_labels_are_human_readable():
    assert provider_label("alphavantage") == "Alpha Vantage" and "yFinance" in provider_label("yfinance")
    assert provider_label("mock") == "sample data" and provider_label(None) == "an earlier lookup"
    assert provider_label("newprovider") == "newprovider"


def test_very_fresh_cache_reads_naturally():
    text = build_notice("cached", "alphavantage", 5)
    assert "only moments old" in text and "about moments" not in text


def test_live_data_has_no_notice_and_every_other_status_has_one():
    assert build_notice(DataStatus.live, "alphavantage") is None
    for status in (DataStatus.cached, DataStatus.stale, DataStatus.sample):
        for kind in ("price", "history", "news"):
            assert build_notice(status, "alphavantage", 600, kind=kind)


def test_notice_wording_per_status():
    cached = build_notice("cached", "alphavantage", 600)
    assert "Cached" in cached and "Alpha Vantage" in cached and "10 minutes" in cached and "not a fresh lookup" in cached
    stale = build_notice("stale", "yfinance", 3 * 3600, kind="history")
    assert "unavailable" in stale and "last saved price history" in stale and "3 hours" in stale
    sample = build_notice("sample", "mock")
    assert "SAMPLE" in sample and "not a real market figure" in sample
    assert "SAMPLE headlines" in build_notice("sample", "sample", kind="news")


# ------------------------------------------------------------------------- Quote self-discloses
def test_live_quote_has_no_notice():
    q = Quote(symbol="VTI", price=1, source="alphavantage")
    assert q.data_status is DataStatus.live and q.notice is None and not q.stale


def test_mock_quote_can_never_exist_without_a_sample_notice():
    q = Quote(symbol="VTI", price=1, source="mock")  # even if a caller forgets every flag
    assert q.data_status is DataStatus.sample and q.stale and "SAMPLE" in q.notice
    assert "SAMPLE" in MockProvider().get_quote("VTI").notice


def test_stale_flag_alone_forces_a_notice():
    q = Quote(symbol="VTI", price=1, source="yfinance", stale=True)
    assert q.data_status is DataStatus.stale and q.notice and "last saved" in q.notice


def test_with_status_uses_given_cache_age_and_sets_flags():
    base = Quote(symbol="VTI", price=1, source="alphavantage")
    cached = base.with_status(DataStatus.cached, 720)
    assert cached.cache_hit and not cached.stale and "12 minutes" in cached.notice and "Alpha Vantage" in cached.notice
    stale = base.with_status(DataStatus.stale, 7200)
    assert stale.cache_hit and stale.stale and "2 hours" in stale.notice
    assert base.notice is None  # original untouched (copy)


def test_naive_as_of_is_handled():
    q = Quote(symbol="VTI", price=1, source="alphavantage", as_of=datetime.now() - timedelta(minutes=5))
    assert q.age_seconds() >= 0


# ------------------------------------------------------------------- service: every path discloses
class Fake:
    is_sample = False

    def __init__(self, name, mode="ok"):
        self.name, self.mode = name, mode

    def supports(self, symbol):
        return True

    def _go(self):
        if self.mode == "down":
            raise ProviderUnavailableError("down")

    def get_quote(self, symbol):
        self._go()
        return Quote(symbol=symbol, price=10.0, source=self.name)

    def get_history(self, symbol, days):
        self._go()
        start = datetime.now(timezone.utc).date() - timedelta(days=60)
        return [PricePoint(day=start + timedelta(days=i), close=10 + i) for i in range(60)]


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "f.db")
    db.init_schema()
    clock = [5_000.0]
    return PersistentCache(db, clock=lambda: clock[0]), clock


def svc(env, *providers):
    return MarketService(list(providers), env[0], MarketConfig(), quote_ttl_seconds=1800)


def test_quote_lifecycle_live_then_cached_then_stale_then_sample(env):
    av, yf = Fake("alphavantage"), Fake("yfinance")
    service = svc(env, av, yf, MockProvider())

    live = service.get_quote("VTI")
    assert live.data_status is DataStatus.live and live.notice is None

    env[1][0] += 600
    cached = service.get_quote("VTI")
    assert cached.data_status is DataStatus.cached and cached.cache_hit
    assert "Cached" in cached.notice and "10 minutes" in cached.notice and "Alpha Vantage" in cached.notice

    env[1][0] += 3 * 3600
    av.mode = yf.mode = "down"
    stale = service.get_quote("VTI")
    assert stale.data_status is DataStatus.stale and stale.stale
    assert "unavailable" in stale.notice and "Alpha Vantage" in stale.notice and "hours" in stale.notice

    sample = service.get_quote("BND")  # never cached, providers down -> sample
    assert sample.data_status is DataStatus.sample and "SAMPLE" in sample.notice and sample.source == "mock"


def test_invariant_no_non_live_quote_ever_lacks_a_notice(env):
    scenarios = [("ok", "ok"), ("down", "ok"), ("down", "down"), ("ok", "down")]
    for i, (a, b) in enumerate(scenarios):
        service = svc(env, Fake("alphavantage", a), Fake("yfinance", b), MockProvider())
        for sym in ("VTI", "BND", "AGG"):
            for _ in range(2):  # second pass exercises the cache path
                q = service.get_quote(sym)
                assert (q.data_status is DataStatus.live) == (q.notice is None), (a, b, sym, q.data_status)
        env[1][0] += 2 * 3600  # let caches age between scenarios


def test_batch_collects_unique_notices(env):
    down = svc(env, Fake("alphavantage", "down"), Fake("yfinance", "down"), MockProvider())
    batch = down.get_quotes(["VTI", "BND", "AGG"])
    assert len(batch.quotes) == 3 and len(batch.notices) == 1 and "SAMPLE" in batch.notices[0]  # deduplicated
    live = svc(env, Fake("alphavantage")).get_quotes(["XYZ"])
    assert live.notices == []


def test_history_and_trend_lifecycle(env):
    av = Fake("alphavantage")
    service = svc(env, av, MockProvider())
    trend, hist = service.get_trend("VTI")
    assert hist.status is DataStatus.live and hist.notice is None and trend.notice is None and trend.data_status is DataStatus.live

    env[1][0] += 3600
    trend, hist = service.get_trend("VTI")
    assert hist.status is DataStatus.cached and "Cached price history" in hist.notice
    assert trend.notice == hist.notice and trend.data_status is DataStatus.cached and trend.notes[0] == hist.notice

    env[1][0] += 24 * 3600
    av.mode = "down"
    trend, hist = service.get_trend("VTI")
    assert hist.status is DataStatus.stale and hist.stale and "last saved price history" in hist.notice

    trend, hist = service.get_trend("BND")
    assert hist.status is DataStatus.sample and "SAMPLE" in trend.notice


def test_history_dataclass_forces_sample_notice_for_mock_source():
    h = History("VTI", [], "mock")
    assert h.status is DataStatus.sample and h.stale and "SAMPLE" in h.notice
    assert History("VTI", [], "alphavantage").notice is None


# ------------------------------------------------------------------------------- news discloses
class Down:
    is_sample = False
    name = "serpapi"

    def search(self, query, limit):
        raise ProviderUnavailableError("down")


class Good:
    is_sample = False
    name = "serpapi"

    def search(self, query, limit):
        return [NewsItem(title="T", url="https://x/t", provider="serpapi")]


def news(env, *providers):
    return NewsService([*providers, SampleNewsProvider()], env[0], ttl_seconds=1800, stale_max_age_seconds=86400, redactor=lambda t: t)


def test_news_lifecycle_live_cached_stale_sample(env):
    good = Good()
    service = news(env, good)
    _, live = service.search("etfs")
    assert live["data_status"] == "live" and live["notice"] is None

    env[1][0] += 900
    _, cached = service.search("etfs")
    assert cached["data_status"] == "cached" and cached["cache_hit"] and "15 minutes" in cached["notice"] and "Google News" in cached["notice"]

    env[1][0] += 3 * 3600
    down = news(env, Down())
    items, stale = down.search("etfs")
    assert items[0].title == "T" and stale["data_status"] == "stale" and "last saved headlines" in stale["notice"]

    items, sample = down.search("never-seen topic")
    assert sample["data_status"] == "sample" and sample["source"] == "sample" and "SAMPLE headlines" in sample["notice"]


def test_negative_cache_path_also_discloses(env):
    class Empty:
        is_sample = False
        name = "serpapi"

        def search(self, query, limit):
            return []

    service = news(env, Empty())
    _, first = service.search("obscure question here")
    _, second = service.search("obscure question here")
    assert first["data_status"] == second["data_status"] == "sample" and second["cache_hit"] and second["notice"]


# --------------------------------------------------------------------- narrated answers keep it
def test_with_data_notices_appends_once_and_skips_when_empty():
    text = "VTI closed at $265."
    assert with_data_notices(text, None) == text and with_data_notices(text, []) == text and with_data_notices(text, [None, ""]) == text
    out = with_data_notices(text, ["Cached data from Alpha Vantage, about 10 minutes old.", "Cached data from Alpha Vantage, about 10 minutes old.", "Live news is unavailable."])
    assert out.count("Cached data") == 1 and "Data note:" in out and "**" not in out and "- Live news is unavailable." in out
    assert with_data_notices(out, ["Cached data from Alpha Vantage, about 10 minutes old.", "Live news is unavailable."]) == out  # idempotent
