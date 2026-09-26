import httpx
import pytest

from src.core.errors import ProviderUnavailableError
from src.core.models import NewsItem
from src.data import news_client
from src.data.cache import PersistentCache, UsageTracker
from src.data.db import Database
from src.data.news_client import keyword_query, NewsService, SampleNewsProvider, SerpApiNewsProvider, YFinanceNewsProvider, build_news_service
from src.utils.rate_limit import QuotaGuard

KEY = "SERPKEY-abcdef0123456789"
SERP_OK = {"news_results": [
    {"title": "Index funds keep winning", "link": "https://ex.com/a", "source": {"name": "Barron's"}, "date": "09/21/2026", "snippet": "Passive beats active."},
    {"stories": [{"title": "Nested story", "link": "https://ex.com/b", "source": {"name": "Reuters"}}]},
    {"title": "no link here"},
    {"link": "https://ex.com/no-title"},
]}


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "n.db")
    d.init_schema()
    return d


def serp(handler, db, **kw):
    guard = QuotaGuard("serpapi", period_limit=100, period_kind="month", usage=UsageTracker(db))
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return SerpApiNewsProvider(KEY, guard, client=client, sleep=lambda s: None, base_delay=0.01, **kw), guard


def ok(payload, status=200):
    return lambda request: httpx.Response(status, json=payload)


# ------------------------------------------------------------------------------------ SerpAPI
def test_serpapi_parses_flat_and_nested_stories_and_skips_incomplete(db):
    provider, guard = serp(ok(SERP_OK), db)
    items = provider.search("index funds", 8)
    assert [i.title for i in items] == ["Index funds keep winning", "Nested story"]
    assert items[0].source == "Barron's" and items[0].snippet == "Passive beats active." and items[0].provider == "serpapi"
    assert guard.remaining() == 99


def test_serpapi_respects_limit_and_sends_expected_params(db):
    seen = {}

    def handler(request):
        seen.update(request.url.params)
        return httpx.Response(200, json=SERP_OK)

    assert len(serp(handler, db)[0].search("etf", 1)) == 1
    assert seen["engine"] == "google_news" and seen["q"] == "etf" and seen["api_key"] == KEY


def test_serpapi_no_results_is_empty_not_an_error(db):
    assert serp(ok({"error": "Google hasn't returned any results for this query."}), db)[0].search("zzz", 5) == []


@pytest.mark.parametrize("status,match", [(401, "rejected"), (403, "rejected"), (404, "HTTP 404")])
def test_serpapi_http_errors(db, status, match):
    with pytest.raises(ProviderUnavailableError, match=match):
        serp(ok({}, status), db)[0].search("x", 5)


def test_serpapi_retries_server_errors_and_never_leaks_key(db):
    hits = []

    def handler(request):
        hits.append(1)
        raise httpx.ConnectError(f"cannot reach {request.url}")

    with pytest.raises(ProviderUnavailableError) as info:
        serp(handler, db, attempts=3)[0].search("x", 5)
    assert len(hits) == 3 and KEY not in str(info.value)
    with pytest.raises(ProviderUnavailableError, match="HTTP 502"):
        serp(ok({}, 502), db, attempts=2)[0].search("x", 5)


def test_serpapi_api_error_bad_json_and_missing_key(db):
    with pytest.raises(ProviderUnavailableError, match="reported an error"):
        serp(ok({"error": "Invalid API key"}), db)[0].search("x", 5)
    with pytest.raises(ProviderUnavailableError, match="invalid JSON"):
        serp(lambda r: httpx.Response(200, content=b"<html>"), db)[0].search("x", 5)
    guard = QuotaGuard("serpapi", period_limit=100, period_kind="month", usage=UsageTracker(db))
    with pytest.raises(ProviderUnavailableError, match="not configured"):
        SerpApiNewsProvider("", guard).search("x", 5)


def test_serpapi_monthly_quota_blocks_further_calls(db):
    guard = QuotaGuard("serpapi", period_limit=1, period_kind="month", usage=UsageTracker(db))
    hits = []
    provider = SerpApiNewsProvider(KEY, guard, client=httpx.Client(transport=httpx.MockTransport(lambda r: (hits.append(1), httpx.Response(200, json=SERP_OK))[1])))
    provider.search("a", 5)
    with pytest.raises(ProviderUnavailableError, match="quota"):
        provider.search("b", 5)
    assert len(hits) == 1


# ---------------------------------------------------------------------------------- yFinance news
class FakeTicker:
    def __init__(self, news=None, error=None):
        self._news, self._error = news, error

    @property
    def news(self):
        if self._error:
            raise self._error
        return self._news


def test_yfinance_news_handles_new_and_old_formats():
    new = {"content": {"title": "New format", "canonicalUrl": {"url": "https://y.com/1"}, "provider": {"displayName": "Yahoo"}, "pubDate": "2026-09-20", "summary": "s"}}
    old = {"title": "Old format", "link": "https://y.com/2", "publisher": "Reuters", "providerPublishTime": 1790000000}
    junk = {"content": {"title": "no url"}}
    items = YFinanceNewsProvider(lambda s: FakeTicker([new, old, junk])).search("AAPL stock outlook", 5)
    assert [i.title for i in items] == ["New format", "Old format"] and items[0].source == "Yahoo" and items[1].source == "Reuters"


@pytest.mark.parametrize("query", ["", "what is an etf", "index funds", "12345", "explain the IRA and ETF basics", "the FED and CPI"])
def test_yfinance_news_requires_a_ticker(query):
    with pytest.raises(ProviderUnavailableError):
        YFinanceNewsProvider(lambda s: FakeTicker([])).search(query, 3)


@pytest.mark.parametrize("query,expected", [("AAPL stock outlook", "AAPL"), ("news about TSLA today", "TSLA"), ("$vti risks", "VTI"),
                                             ("thoughts on BRK-B?", "BRK-B"), ("$a", "A")])
def test_yfinance_news_finds_explicit_tickers(query, expected):
    seen = []
    provider = YFinanceNewsProvider(lambda s: (seen.append(s), FakeTicker([]))[1])
    provider.search(query, 3)
    assert seen == [expected]


def test_yfinance_news_failure_is_unavailable():
    with pytest.raises(ProviderUnavailableError, match="TimeoutError"):
        YFinanceNewsProvider(lambda s: FakeTicker(error=TimeoutError("x"))).search("AAPL", 3)


def test_sample_news_is_labelled():
    items = SampleNewsProvider().search("anything", 2)
    assert len(items) == 2 and all(i.provider == "sample" and i.title.startswith("Sample") for i in items)


# ---------------------------------------------------------------------------------------- service
class Recorder:
    is_sample = False

    def __init__(self, name, items=None, error=None):
        self.name, self.items, self.error, self.queries = name, items or [], error, []

    def search(self, query, limit):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.items


def item(title, provider):
    return NewsItem(title=title, url=f"https://x.com/{title}", provider=provider)


def news(db, *providers, redactor=lambda t: t, clock=None):
    cache = PersistentCache(db, clock=clock) if clock else PersistentCache(db)
    return NewsService(list(providers) + [SampleNewsProvider()], cache, ttl_seconds=1800, stale_max_age_seconds=86400, redactor=redactor)


def test_query_is_redacted_before_it_reaches_any_provider(db):
    rec = Recorder("serpapi", [item("A", "serpapi")])
    news(db, rec, redactor=lambda t: t.replace("john@example.com", "<EMAIL_ADDRESS>")).search("news for john@example.com about VTI")
    assert rec.queries == ["news for <EMAIL_ADDRESS> about VTI"]


def test_real_pii_redactor_end_to_end(db):
    rec = Recorder("serpapi", [item("A", "serpapi")])
    news(db, rec, redactor=None).search("headlines about my SSN 536-90-4399 and john.smith@example.com")
    assert "536-90-4399" not in rec.queries[0] and "john.smith@example.com" not in rec.queries[0]


def test_redactor_failure_fails_closed(db):
    def boom(_):
        raise RuntimeError("model missing")

    rec = Recorder("serpapi", [item("A", "serpapi")])
    with pytest.raises(RuntimeError):
        news(db, rec, redactor=boom).search("anything")
    assert rec.queries == []  # nothing left the machine


def test_primary_then_cache_hit(db):
    rec = Recorder("serpapi", [item("A", "serpapi")])
    svc = news(db, rec)
    items, meta = svc.search("index funds")
    assert [i.title for i in items] == ["A"]
    assert (meta["source"], meta["stale"], meta["cache_hit"], meta["data_status"], meta["notice"]) == ("serpapi", False, False, "live", None)
    items2, meta2 = svc.search("INDEX   funds ")
    assert meta2["cache_hit"] and len(rec.queries) == 1  # normalized query hits the same cache entry


def test_falls_back_to_yfinance_news_when_serpapi_fails_or_is_empty(db):
    down = Recorder("serpapi", error=ProviderUnavailableError("quota"))
    yf = Recorder("yfinance", [item("Y", "yfinance")])
    assert news(db, down, yf).search("AAPL")[1]["source"] == "yfinance"
    empty = Recorder("serpapi", [])
    assert news(db, empty, yf).search("MSFT")[1]["source"] == "yfinance"


def test_unexpected_provider_bug_is_contained(db):
    bug = Recorder("serpapi", error=RuntimeError("bug"))
    yf = Recorder("yfinance", [item("Y", "yfinance")])
    assert news(db, bug, yf).search("AAPL")[1]["source"] == "yfinance"


def test_all_down_serves_last_good_then_sample(db):
    now = [1000.0]
    good = Recorder("serpapi", [item("Cached", "serpapi")])
    svc = news(db, good, clock=lambda: now[0])
    svc.search("etfs")
    now[0] += 7200  # cache expired, providers now down
    good.error = ProviderUnavailableError("down")
    items, meta = svc.search("etfs")
    assert items[0].title == "Cached" and meta["stale"] and meta["cache_hit"]
    fresh_items, fresh_meta = news(db, Recorder("serpapi", error=ProviderUnavailableError("down"))).search("brand new query")
    assert fresh_meta["source"] == "sample" and fresh_meta["stale"] and fresh_items[0].title.startswith("Sample")


def test_empty_and_long_queries_are_normalized(db):
    rec = Recorder("serpapi", [item("A", "serpapi")])
    svc = news(db, rec)
    svc.search("   ")
    svc.search("x" * 500)
    assert rec.queries[0] == "stock market" and len(rec.queries[1]) == 200


def test_no_sample_provider_and_nothing_available_returns_empty(db):
    svc = NewsService([Recorder("serpapi", error=ProviderUnavailableError("down"))], PersistentCache(db), ttl_seconds=60, stale_max_age_seconds=60, redactor=lambda t: t)
    items, meta = svc.search("x")
    assert items == [] and meta["source"] is None and meta["stale"] and meta["data_status"] == "sample" and "SAMPLE" in meta["notice"]


def test_factory_wires_serpapi_when_key_present_and_skips_otherwise(tmp_path):
    original = news_client.settings.serpapi_api_key
    try:
        object.__setattr__(news_client.settings, "serpapi_api_key", "KEY")
        assert [p.name for p in build_news_service(Database(tmp_path / "a.db")).providers] == ["serpapi", "yfinance", "sample"]
        object.__setattr__(news_client.settings, "serpapi_api_key", "")
        assert [p.name for p in build_news_service(Database(tmp_path / "b.db")).providers] == ["yfinance", "sample"]
    finally:
        object.__setattr__(news_client.settings, "serpapi_api_key", original)


# ---------------------------------------------------------------------- keywords and negative caching
@pytest.mark.parametrize("question,expected", [
    ("index funds vs ETFs for beginners", "index funds ETFs beginners"),
    ("What is the difference between a Roth IRA and a 401k?", "Roth IRA 401k"),
    ("how should I invest in AAPL", "invest AAPL"),
    ("the and of", ""),
])
def test_keyword_query(question, expected):
    assert keyword_query(question) == expected


def test_keyword_query_caps_word_count():
    assert len(keyword_query("alpha bravo charlie delta echo foxtrot golf hotel").split()) == 6


class Scripted:
    """Returns [] for the natural-language query and results for the keyword version."""

    is_sample = False
    name = "serpapi"

    def __init__(self, by_query):
        self.by_query, self.queries = by_query, []

    def search(self, query, limit):
        self.queries.append(query)
        return self.by_query.get(query, [])


def test_empty_natural_language_query_is_retried_with_keywords(db):
    prov = Scripted({"index funds ETFs beginners": [item("Found", "serpapi")]})
    items, meta = news(db, prov).search("index funds vs ETFs for beginners")
    assert [i.title for i in items] == ["Found"] and meta["source"] == "serpapi"
    assert prov.queries == ["index funds vs ETFs for beginners", "index funds ETFs beginners"]


def test_no_results_are_cached_so_quota_is_not_spent_twice(db):
    prov = Scripted({})
    svc = news(db, prov)
    first_items, first_meta = svc.search("index funds vs ETFs for beginners")
    assert first_meta["source"] == "sample" and len(prov.queries) == 2  # original + keyword retry
    again_items, again_meta = svc.search("index funds vs ETFs for beginners")
    assert again_meta["source"] == "sample" and again_meta["cache_hit"] and len(prov.queries) == 2  # no new calls


def test_negative_cache_expires_and_never_erases_a_last_good_value(db):
    now = [1000.0]
    prov = Scripted({"etfs": [item("Good", "serpapi")]})
    svc = news(db, prov, clock=lambda: now[0])
    svc.search("etfs")
    now[0] += 7200  # cached items expired
    prov.by_query = {}  # provider now returns nothing for this query
    items, meta = svc.search("etfs")
    assert items[0].title == "Good" and meta["stale"]  # last-good survives alongside the "empty" note
    calls = len(prov.queries)
    svc.search("etfs")
    assert len(prov.queries) == calls  # negative cache suppressed the re-query
    now[0] += 7200
    prov.by_query = {"etfs": [item("Fresh", "serpapi")]}
    assert svc.search("etfs")[0][0].title == "Fresh"


def test_errors_are_not_negatively_cached(db):
    prov = Recorder("serpapi", error=ProviderUnavailableError("down"))
    svc = news(db, prov)
    svc.search("some topic")
    svc.search("some topic")
    assert len(prov.queries) == 2  # a provider outage must be retried next time


# ------------------------------------------------------------------------------- ordering (latest first)
def dated_item(title, published):
    return NewsItem(title=title, url=f"https://x.com/{title}", provider="serpapi", published=published)


def test_items_are_sorted_by_published_date_descending(db):
    old = dated_item("Old", "09/20/2026, 08:00 AM, +0000 UTC")
    newest = dated_item("Newest", "09/22/2026, 08:00 AM, +0000 UTC")
    mid = dated_item("Mid", "09/21/2026, 08:00 AM, +0000 UTC")
    items, _ = news(db, Recorder("serpapi", [old, newest, mid])).search("index funds")
    assert [i.title for i in items] == ["Newest", "Mid", "Old"]


def test_unparseable_dates_sort_last_preserving_original_order(db):
    known = dated_item("Known", "09/22/2026, 08:00 AM, +0000 UTC")
    a, b = dated_item("A", None), dated_item("B", None)
    items, _ = news(db, Recorder("serpapi", [a, known, b])).search("topic")
    assert [i.title for i in items] == ["Known", "A", "B"]


def test_cached_and_stale_reads_stay_sorted(db):
    now = [1000.0]
    old = dated_item("Old", "09/20/2026, 08:00 AM, +0000 UTC")
    newest = dated_item("Newest", "09/22/2026, 08:00 AM, +0000 UTC")
    prov = Recorder("serpapi", [old, newest])
    svc = news(db, prov, clock=lambda: now[0])
    svc.search("index funds")
    cached, meta = svc.search("index funds")
    assert [i.title for i in cached] == ["Newest", "Old"] and meta["cache_hit"]
    now[0] += 7200  # past the TTL, and the provider is now down: falls back to the stale last-good copy
    prov.error = ProviderUnavailableError("down")
    stale, meta2 = svc.search("index funds")
    assert [i.title for i in stale] == ["Newest", "Old"] and meta2["stale"]


@pytest.mark.parametrize("value,expected", [
    ("09/22/2026, 08:00 AM, +0000 UTC", (2026, 9, 22, 8)),
    ("2026-09-22T08:00:00Z", (2026, 9, 22, 8)),
    ("Mon, 22 Sep 2026 08:00:00 GMT", (2026, 9, 22, 8)),
])
def test_parse_published_recognizes_provider_formats(value, expected):
    dt = news_client._parse_published(value)
    assert (dt.year, dt.month, dt.day, dt.hour) == expected


def test_parse_published_epoch_seconds():
    dt = news_client._parse_published("1758528000")
    assert dt is not None and dt.year == 2025


@pytest.mark.parametrize("value", [None, "", "   ", "not a date"])
def test_parse_published_returns_none_for_junk(value):
    assert news_client._parse_published(value) is None


def test_parse_published_falls_back_to_rfc822_for_a_naive_date():
    dt = news_client._parse_published("22 Sep 2026 08:00:00")  # no weekday/timezone: none of the strict formats match
    assert (dt.year, dt.month, dt.day, dt.hour) == (2026, 9, 22, 8) and dt.tzinfo is not None


def test_parse_published_rejects_an_out_of_range_epoch():
    assert news_client._parse_published("9" * 30) is None
