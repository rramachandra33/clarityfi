from datetime import date, datetime, timedelta

import httpx
import pandas as pd
import pytest

from src.core.errors import ProviderUnavailableError, SymbolNotFoundError
from src.data.cache import UsageTracker
from src.data.db import Database
from src.data.market_client import AlphaVantageProvider, MockProvider, YFinanceProvider, MOCK_BASE_PRICES
from src.utils.rate_limit import QuotaGuard

KEY = "TESTKEY1234567890"
GOOD_QUOTE = {"Global Quote": {
    "01. symbol": "AAPL", "05. price": "336.1300", "07. latest trading day": "2026-09-18",
    "08. previous close": "337.0000", "09. change": "-0.8700", "10. change percent": "-0.2582%"}}


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "m.db")
    d.init_schema()
    return d


def av(handler, db, **kw):
    guard = QuotaGuard("alphavantage", per_minute=5, period_limit=25, usage=UsageTracker(db))
    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleeps = kw.pop("sleeps", [])
    kw.setdefault("min_interval", 0)  # pacing is tested separately
    return AlphaVantageProvider(KEY, guard, client=client, sleep=sleeps.append, base_delay=0.01, **kw), guard


def respond(payload, status=200):
    return lambda request: httpx.Response(status, json=payload)


# -------------------------------------------------------------------------------- Alpha Vantage
def test_av_parses_quote(db):
    provider, _ = av(respond(GOOD_QUOTE), db)
    q = provider.get_quote("AAPL")
    assert (q.symbol, q.price, q.previous_close, q.change_pct, q.source) == ("AAPL", 336.13, 337.0, -0.2582, "alphavantage")
    assert q.trading_day == date(2026, 9, 18) and not q.stale


def test_av_sends_key_and_symbol_as_params(db):
    seen = {}

    def handler(request):
        seen.update(request.url.params)
        return httpx.Response(200, json=GOOD_QUOTE)

    av(handler, db)[0].get_quote("AAPL")
    assert seen["function"] == "GLOBAL_QUOTE" and seen["symbol"] == "AAPL" and seen["apikey"] == KEY


@pytest.mark.parametrize("payload", [{"Global Quote": {}}, {"Error Message": "Invalid API call"}, {}])
def test_av_unknown_symbol(db, payload):
    with pytest.raises(SymbolNotFoundError):
        av(respond(payload), db)[0].get_quote("ZZZZ")


def test_av_daily_quota_message_exhausts_budget_and_stops_calling(db):
    hits = []

    def handler(request):
        hits.append(1)
        return httpx.Response(200, json={"Information": DAILY})

    provider, guard = av(handler, db)
    with pytest.raises(ProviderUnavailableError):
        provider.get_quote("AAPL")
    assert guard.remaining() == 0
    with pytest.raises(ProviderUnavailableError, match="quota or rate limit"):
        provider.get_quote("MSFT")
    assert len(hits) == 1  # the second call never reached the network


BURST = ("Thank you for using Alpha Vantage! Please consider spreading out your free API requests more sparingly "
         "(1 request per second). You may subscribe to any of the premium plans to lift the free key rate limit (25 requests per day).")
DAILY = "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day. Please subscribe to premium plans."


def test_av_burst_message_is_retried_and_does_not_exhaust_the_day(db):
    hits = []

    def handler(request):
        hits.append(1)
        return httpx.Response(200, json={"Information": BURST} if len(hits) == 1 else GOOD_QUOTE)

    provider, guard = av(handler, db)
    assert provider.get_quote("AAPL").price == 336.13  # second attempt succeeded
    assert len(hits) == 2 and guard.remaining() == 23


def test_av_persistent_burst_message_never_marks_the_day_exhausted(db):
    provider, guard = av(respond({"Information": BURST}), db, attempts=2)
    with pytest.raises(ProviderUnavailableError, match="burst"):
        provider.get_quote("AAPL")
    assert guard.remaining() == 23  # only the two real attempts were counted, not the whole day


def test_av_requests_are_paced_to_stay_under_one_per_second(db):
    sleeps, now = [], [100.0]
    guard = QuotaGuard("alphavantage", period_limit=25, usage=UsageTracker(db))
    client = httpx.Client(transport=httpx.MockTransport(respond(GOOD_QUOTE)))
    provider = AlphaVantageProvider(KEY, guard, client=client, sleep=lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)),
                                    clock=lambda: now[0], min_interval=1.1)
    provider.get_quote("AAPL")
    assert sleeps == []  # first call is immediate
    now[0] += 0.3
    provider.get_quote("MSFT")
    assert sleeps == [pytest.approx(0.8)]  # waited out the remainder of the 1.1s window
    now[0] += 5
    provider.get_quote("VTI")
    assert len(sleeps) == 1  # plenty of time had passed, no wait


def test_av_server_errors_are_retried_then_reported(db):
    hits, sleeps = [], []

    def handler(request):
        hits.append(1)
        return httpx.Response(503)

    provider, _ = av(handler, db, sleeps=sleeps, attempts=3)
    with pytest.raises(ProviderUnavailableError, match="HTTP 503"):
        provider.get_quote("AAPL")
    assert len(hits) == 3 and len(sleeps) == 2


def test_av_recovers_after_transient_failure(db):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500) if len(calls) == 1 else httpx.Response(200, json=GOOD_QUOTE)

    assert av(handler, db)[0].get_quote("AAPL").price == 336.13


def test_av_timeouts_retry_and_never_leak_the_key(db):
    def handler(request):
        raise httpx.ConnectTimeout(f"timed out for {request.url}")  # message contains the key-bearing URL

    provider, _ = av(handler, db, attempts=2)
    with pytest.raises(ProviderUnavailableError) as info:
        provider.get_quote("AAPL")
    assert KEY not in str(info.value) and "apikey" not in str(info.value).lower()


def test_av_each_attempt_consumes_quota(db):
    provider, guard = av(respond({}, 500), db, attempts=3)
    with pytest.raises(ProviderUnavailableError):
        provider.get_quote("AAPL")
    assert guard.remaining() == 22


def test_av_non_200_bad_json_and_bad_payloads(db):
    with pytest.raises(ProviderUnavailableError, match="HTTP 403"):
        av(respond({}, 403), db)[0].get_quote("AAPL")
    with pytest.raises(ProviderUnavailableError, match="invalid JSON"):
        av(lambda r: httpx.Response(200, content=b"<html>"), db)[0].get_quote("AAPL")
    with pytest.raises(ProviderUnavailableError, match="unexpected payload"):
        av(lambda r: httpx.Response(200, json=[1, 2]), db)[0].get_quote("AAPL")
    bad = {"Global Quote": {"05. price": "not-a-number"}}
    with pytest.raises(ProviderUnavailableError, match="malformed"):
        av(respond(bad), db)[0].get_quote("AAPL")


def test_av_without_key_is_unavailable_without_network(db):
    guard = QuotaGuard("alphavantage", period_limit=25, usage=UsageTracker(db))
    provider = AlphaVantageProvider("", guard, client=httpx.Client(transport=httpx.MockTransport(lambda r: pytest.fail("no call"))))
    with pytest.raises(ProviderUnavailableError, match="not configured"):
        provider.get_quote("AAPL")


def test_av_does_not_support_crypto_pairs(db):
    provider = av(respond({}), db)[0]
    assert provider.supports("AAPL") and provider.supports("BRK-B")
    assert not provider.supports("BTC-USD") and not provider.supports("ETH-EUR")


def test_av_history_parses_and_filters_by_days(db):
    today = date.today()
    series = {(today - timedelta(days=d)).isoformat(): {"4. close": str(100 + d)} for d in (1, 5, 30, 200)}
    provider, _ = av(respond({"Time Series (Daily)": series}), db)
    points = provider.get_history("VTI", days=60)
    assert [p.close for p in points] == [130.0, 105.0, 101.0]  # oldest first, 200-day-old row dropped
    assert points == sorted(points, key=lambda p: p.day)


def test_av_history_errors(db):
    with pytest.raises(SymbolNotFoundError):
        av(respond({"Meta Data": {}}), db)[0].get_history("ZZZ", 30)
    with pytest.raises(ProviderUnavailableError, match="malformed"):
        av(respond({"Time Series (Daily)": {"2026-09-01": {"nope": "1"}}}), db)[0].get_history("VTI", 500)


# ---------------------------------------------------------------------------------------- yFinance
class FakeTicker:
    def __init__(self, info=None, frame=None, fail=None):
        self.fast_info, self._frame, self._fail = info or {}, frame, fail

    def history(self, **kwargs):
        if self._fail:
            raise self._fail
        return self._frame


def yf_provider(ticker, attempts=2):
    sleeps = []
    return YFinanceProvider(ticker_factory=lambda s: ticker, attempts=attempts, base_delay=0.01, sleep=sleeps.append), sleeps


def test_yf_quote():
    p, _ = yf_provider(FakeTicker({"last_price": 337.66, "previous_close": 334.8}))
    q = p.get_quote("AAPL")
    assert q.price == 337.66 and q.previous_close == 334.8 and q.change_pct == pytest.approx(0.8542, abs=1e-3)
    assert q.source == "yfinance" and p.supports("BTC-USD")


@pytest.mark.parametrize("info", [{"last_price": None, "previous_close": 1}, {"last_price": float("nan"), "previous_close": 1},
                                  {"last_price": 0, "previous_close": 1}])
def test_yf_missing_price_means_symbol_not_found(info):
    with pytest.raises(SymbolNotFoundError):
        yf_provider(FakeTicker(info))[0].get_quote("ZZZZ")


def test_yf_network_errors_retry_then_unavailable():
    class Exploding(FakeTicker):
        @property
        def fast_info(self):
            raise ConnectionError("yahoo down")

        @fast_info.setter
        def fast_info(self, value):
            pass

    p, sleeps = yf_provider(Exploding(), attempts=3)
    with pytest.raises(ProviderUnavailableError, match="ConnectionError"):
        p.get_quote("AAPL")
    assert len(sleeps) == 2


def test_yf_history():
    idx = pd.to_datetime(["2026-09-15", "2026-09-16", "2026-09-17"])
    frame = pd.DataFrame({"Close": [10.0, float("nan"), 12.5]}, index=idx)
    points = yf_provider(FakeTicker(frame=frame))[0].get_history("VTI", 30)
    assert [(p.day, p.close) for p in points] == [(date(2026, 9, 15), 10.0), (date(2026, 9, 17), 12.5)]


def test_yf_history_empty_or_failing():
    with pytest.raises(SymbolNotFoundError):
        yf_provider(FakeTicker(frame=pd.DataFrame({"Close": []})))[0].get_history("ZZZ", 30)
    with pytest.raises(ProviderUnavailableError):
        yf_provider(FakeTicker(fail=TimeoutError("slow")))[0].get_history("VTI", 30)
    with pytest.raises(SymbolNotFoundError):
        yf_provider(FakeTicker(frame=pd.DataFrame({"Close": [float("nan")]}, index=pd.to_datetime(["2026-01-01"]))))[0].get_history("X", 5)


# -------------------------------------------------------------------------------------- sample data
def test_mock_quote_is_deterministic_flagged_and_bounded():
    m = MockProvider()
    a, b = m.get_quote("VTI"), m.get_quote("VTI")
    assert (a.price, a.change_pct, a.trading_day) == (b.price, b.change_pct, b.trading_day)  # as_of differs
    assert a.stale and a.source == "mock"
    assert abs(a.price / MOCK_BASE_PRICES["VTI"] - 1) <= 0.0151
    assert m.is_sample and m.supports("VTI") and not m.supports("NOPE")
    with pytest.raises(SymbolNotFoundError):
        m.get_quote("NOPE")


def test_mock_history_shape():
    points = MockProvider().get_history("BND", 90)
    assert 50 <= len(points) <= 70 and points == sorted(points, key=lambda p: p.day)
    assert all(p.day.weekday() < 5 for p in points)
    assert points[-1].close == pytest.approx(MOCK_BASE_PRICES["BND"], abs=0.01)
    assert MockProvider().get_history("BND", 90) == points  # deterministic
    with pytest.raises(SymbolNotFoundError):
        MockProvider().get_history("NOPE", 30)
