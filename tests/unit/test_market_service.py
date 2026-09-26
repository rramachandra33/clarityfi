from datetime import date, timedelta

import pytest

from src.core.config import MarketConfig
from src.core.errors import ProviderUnavailableError, SymbolNotFoundError
from src.core.models import PricePoint, Quote
from src.data import market_client
from src.data.cache import PersistentCache
from src.data.db import Database
from src.data.market_client import MarketService, MockProvider, build_market_service


class Fake:
    """Configurable provider double that counts calls."""

    is_sample = False

    def __init__(self, name, *, price=100.0, mode="ok", supports=True):
        self.name, self.price, self.mode, self._supports = name, price, mode, supports
        self.quote_calls = self.history_calls = 0

    def supports(self, symbol):
        return self._supports

    def _maybe_fail(self):
        if self.mode == "down":
            raise ProviderUnavailableError(f"{self.name} is down")
        if self.mode == "notfound":
            raise SymbolNotFoundError("nope")
        if self.mode == "bug":
            raise RuntimeError("unexpected provider bug")

    def get_quote(self, symbol):
        self.quote_calls += 1
        self._maybe_fail()
        return Quote(symbol=symbol, price=self.price, source=self.name)

    def get_history(self, symbol, days):
        self.history_calls += 1
        self._maybe_fail()
        start = date.today() - timedelta(days=90)
        return [PricePoint(day=start + timedelta(days=i), close=self.price + i * 0.1) for i in range(90)]


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "svc.db")
    db.init_schema()
    clock = [10_000.0]
    cache = PersistentCache(db, clock=lambda: clock[0])
    return db, cache, clock


def service(env, *providers, ttl=1800):
    _, cache, _ = env
    return MarketService(list(providers), cache, MarketConfig(), quote_ttl_seconds=ttl)


def test_primary_provider_is_used_first_and_result_is_cached(env):
    av, yf = Fake("alphavantage", price=101), Fake("yfinance", price=102)
    svc = service(env, av, yf, MockProvider())
    q = svc.get_quote("aapl")
    assert (q.symbol, q.price, q.source, q.cache_hit, q.stale) == ("AAPL", 101, "alphavantage", False, False)
    again = svc.get_quote("AAPL")
    assert again.cache_hit and again.source == "alphavantage" and av.quote_calls == 1 and yf.quote_calls == 0


def test_cache_expires_after_ttl(env):
    av = Fake("alphavantage")
    svc = service(env, av)
    svc.get_quote("VTI")
    env[2][0] += 1801
    svc.get_quote("VTI")
    assert av.quote_calls == 2


def test_falls_back_to_yfinance_when_alpha_vantage_is_down(env):
    av, yf = Fake("alphavantage", mode="down"), Fake("yfinance", price=55)
    q = service(env, av, yf, MockProvider()).get_quote("VTI")
    assert q.source == "yfinance" and q.price == 55 and not q.stale
    assert av.quote_calls == 1 and yf.quote_calls == 1


def test_falls_back_when_alpha_vantage_lacks_the_symbol_or_does_not_support_it(env):
    svc = service(env, Fake("alphavantage", mode="notfound"), Fake("yfinance", price=7))
    assert svc.get_quote("XYZ").source == "yfinance"
    skip = Fake("alphavantage", supports=False)
    assert service(env, skip, Fake("yfinance", price=9)).get_quote("BTC-USD").source == "yfinance" and skip.quote_calls == 0


def test_unexpected_provider_bug_does_not_break_the_chain(env):
    q = service(env, Fake("alphavantage", mode="bug"), Fake("yfinance", price=3)).get_quote("VTI")
    assert q.source == "yfinance"


def test_both_down_serves_last_good_cache_flagged_stale(env):
    av, yf = Fake("alphavantage", price=200), Fake("yfinance")
    svc = service(env, av, yf, MockProvider())
    svc.get_quote("VTI")
    env[2][0] += 3 * 3600  # cache expired
    av.mode = yf.mode = "down"
    q = svc.get_quote("VTI")
    assert q.stale and q.cache_hit and q.source == "alphavantage" and q.price == 200


def test_stale_cache_older_than_limit_is_ignored_and_mock_is_used(env):
    av = Fake("alphavantage")
    svc = service(env, av, MockProvider())
    svc.get_quote("VTI")
    env[2][0] += 8 * 24 * 3600  # older than the 7-day stale window
    av.mode = "down"
    q = svc.get_quote("VTI")
    assert q.source == "mock" and q.stale


def test_both_down_and_no_cache_uses_mock_which_is_never_cached(env):
    _, cache, _ = env
    svc = service(env, Fake("alphavantage", mode="down"), Fake("yfinance", mode="down"), MockProvider())
    q = svc.get_quote("VTI")
    assert q.source == "mock" and q.stale and not q.cache_hit
    assert cache.get_stale("quote:VTI", 10**9) is None  # sample data must not become "last-good"


def test_both_down_unknown_symbol_reports_unavailable(env):
    svc = service(env, Fake("alphavantage", mode="down"), Fake("yfinance", mode="down"), MockProvider())
    with pytest.raises(ProviderUnavailableError, match="No market data"):
        svc.get_quote("QQQQZ")


def test_symbol_unknown_everywhere_is_not_found_and_never_mocked(env):
    svc = service(env, Fake("alphavantage", mode="notfound"), Fake("yfinance", mode="notfound"), MockProvider())
    with pytest.raises(SymbolNotFoundError):
        svc.get_quote("VTI")  # even a symbol the mock knows: providers were up and said no


def test_not_found_plus_outage_falls_through_to_sample(env):
    svc = service(env, Fake("alphavantage", mode="notfound"), Fake("yfinance", mode="down"), MockProvider())
    assert svc.get_quote("VTI").source == "mock"


@pytest.mark.parametrize("bad", ["", "  ", "not a ticker", "TOOLONGSYMBOL", "VTI;DROP", None, 5])
def test_invalid_symbols_rejected_before_any_provider_call(env, bad):
    av = Fake("alphavantage")
    with pytest.raises(SymbolNotFoundError):
        service(env, av).get_quote(bad)
    assert av.quote_calls == 0


def test_get_quotes_dedupes_collects_errors_and_caps(env):
    svc = service(env, Fake("alphavantage", price=10), Fake("yfinance", mode="down"))
    batch = svc.get_quotes(["vti", "VTI", "$vti", "bad symbol", "MSFT"])
    assert list(batch.quotes) == ["VTI", "MSFT"] and "bad symbol" in batch.errors
    many = svc.get_quotes([f"T{chr(65 + i)}" for i in range(26)] + ["EXTRA"])
    assert len(many.quotes) == 25 and "_limit" in many.errors


def test_get_quotes_records_per_symbol_failures(env):
    svc = service(env, Fake("alphavantage", mode="notfound"), Fake("yfinance", mode="notfound"))
    batch = svc.get_quotes(["AAA", "BBB"])
    assert not batch.quotes and set(batch.errors) == {"AAA", "BBB"}


def test_history_chain_cache_and_stale(env):
    av, yf = Fake("alphavantage", price=50), Fake("yfinance")
    svc = service(env, av, yf, MockProvider())
    h = svc.get_history("vti")
    assert h.source == "alphavantage" and len(h.points) == 90 and not h.stale
    svc.get_history("VTI")
    assert av.history_calls == 1  # cached
    env[2][0] += 24 * 3600
    av.mode = yf.mode = "down"
    stale = svc.get_history("VTI")
    assert stale.stale and stale.source == "alphavantage" and len(stale.points) == 90


def test_history_falls_back_to_yfinance_then_mock(env):
    assert service(env, Fake("alphavantage", mode="down"), Fake("yfinance"), MockProvider()).get_history("BND").source == "yfinance"
    h = service(env, Fake("alphavantage", mode="down"), Fake("yfinance", mode="down"), MockProvider()).get_history("AGG")
    assert h.source == "mock" and h.stale  # AGG was never cached, so nothing "last-good" exists
    with pytest.raises(SymbolNotFoundError):
        service(env, Fake("alphavantage", mode="notfound")).get_history("ZZZ")
    with pytest.raises(SymbolNotFoundError):
        service(env, Fake("alphavantage")).get_history("bad symbol")


def test_trend_uses_history_and_labels_sample_data(env):
    svc = service(env, Fake("alphavantage", price=100), MockProvider())
    trend, history = svc.get_trend("VTI")
    assert trend.symbol == "VTI" and trend.points == 90 and trend.trend_label == "Uptrend" and history.source == "alphavantage"
    down = service(env, Fake("alphavantage", mode="down"), MockProvider())
    sample_trend, sample_hist = down.get_trend("BND")
    assert sample_hist.source == "mock" and "SAMPLE data" in sample_trend.notes[0] and sample_trend.notice == sample_hist.notice


def test_status_reports_health_and_quota(env):
    svc = service(env, Fake("alphavantage", mode="down"), Fake("yfinance"), MockProvider())
    svc.get_quote("VTI")
    by_name = {s["provider"]: s for s in svc.status()}
    assert by_name["alphavantage"]["ok"] is False and "down" in by_name["alphavantage"]["last_error"]
    assert by_name["yfinance"]["ok"] is True and by_name["mock"]["sample"] is True
    assert by_name["yfinance"]["quota_remaining"] is None


# ------------------------------------------------------------------------------------- factory
def test_factory_builds_providers_in_configured_priority_order(tmp_path):
    original = market_client.settings.alphavantage_api_key
    object.__setattr__(market_client.settings, "alphavantage_api_key", "KEY")
    try:
        svc = build_market_service(Database(tmp_path / "f.db"))
    finally:
        object.__setattr__(market_client.settings, "alphavantage_api_key", original)
    assert [p.name for p in svc.providers] == ["alphavantage", "yfinance", "mock"]
    assert svc.providers[0].guard.remaining() == 25 and svc.quote_ttl == 1800


def test_factory_skips_alpha_vantage_without_a_key_and_ignores_unknown(tmp_path):
    original = market_client.settings.alphavantage_api_key
    object.__setattr__(market_client.settings, "alphavantage_api_key", "")
    try:
        svc = build_market_service(Database(tmp_path / "g.db"), config=MarketConfig(providers=["alphavantage", "bogus", "yfinance", "mock"]))
    finally:
        object.__setattr__(market_client.settings, "alphavantage_api_key", original)
    assert [p.name for p in svc.providers] == ["yfinance", "mock"]
