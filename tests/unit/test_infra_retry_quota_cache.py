import threading
from datetime import datetime, timezone

import pytest

from src.data.cache import PersistentCache, UsageTracker
from src.data.db import Database
from src.utils.rate_limit import QuotaGuard
from src.utils.retry import RetryableError, retry_call


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "infra.db")
    d.init_schema()
    return d


# ----------------------------------------------------------------------------------- retry
def test_retry_succeeds_after_transient_failures_with_exponential_backoff():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RetryableError("timeout")
        return "ok"

    assert retry_call(flaky, attempts=4, base_delay=1, jitter=0, sleep=sleeps.append) == "ok"
    assert sleeps == [1, 2]  # 1s then 2s


def test_retry_gives_up_and_raises_last_error():
    sleeps = []
    with pytest.raises(RetryableError, match="down"):
        retry_call(lambda: (_ for _ in ()).throw(RetryableError("down")), attempts=3, base_delay=0.1, jitter=0, sleep=sleeps.append)
    assert len(sleeps) == 2


def test_retry_does_not_retry_other_exceptions():
    calls = []

    def boom():
        calls.append(1)
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        retry_call(boom, attempts=5, sleep=lambda s: None)
    assert calls == [1]


def test_retry_delay_is_capped_and_jittered():
    sleeps = []
    with pytest.raises(RetryableError):
        retry_call(lambda: (_ for _ in ()).throw(RetryableError("x")), attempts=6, base_delay=4, max_delay=5, jitter=0, sleep=sleeps.append)
    assert max(sleeps) == 5
    jittered = []
    with pytest.raises(RetryableError):
        retry_call(lambda: (_ for _ in ()).throw(RetryableError("x")), attempts=3, base_delay=1, jitter=0.5, sleep=jittered.append)
    assert 0.5 <= jittered[0] <= 1.5


def test_retry_rejects_zero_attempts():
    with pytest.raises(ValueError):
        retry_call(lambda: 1, attempts=0)


# ------------------------------------------------------------------------------- persistent cache
def test_cache_ttl_and_stale_reads(db):
    now = [1000.0]
    cache = PersistentCache(db, clock=lambda: now[0])
    assert cache.get("k", 60) is None and cache.get_stale("k", 999) is None
    cache.set("k", {"a": 1, "when": datetime(2026, 1, 1, tzinfo=timezone.utc)})
    assert cache.get("k", 60)["a"] == 1
    now[0] += 61
    assert cache.get("k", 60) is None  # expired for fresh reads
    value, age = cache.get_stale("k", 3600)
    assert value["a"] == 1 and age == 61
    assert cache.get_stale("k", 30) is None  # too old even for stale
    cache.set("k", {"a": 2})  # overwrite refreshes the timestamp
    assert cache.get("k", 60)["a"] == 2


def test_cache_delete_prefix(db):
    cache = PersistentCache(db)
    for k in ("quote:A", "quote:B", "news:x"):
        cache.set(k, 1)
    assert cache.delete_prefix("quote:") == 2
    assert cache.get("news:x", 60) == 1


def test_cache_survives_new_instance(db):
    PersistentCache(db).set("persist", [1, 2, 3])
    assert PersistentCache(db).get("persist", 60) == [1, 2, 3]


# --------------------------------------------------------------------------------- usage / quota
def test_usage_tracker_limits_and_periods(db):
    clock = [datetime(2026, 9, 21, 12, tzinfo=timezone.utc)]
    usage = UsageTracker(db, now=lambda: clock[0])
    assert [usage.try_consume("av", 2) for _ in range(3)] == [True, True, False]
    assert usage.get("av") == 2
    clock[0] = datetime(2026, 9, 22, 0, 1, tzinfo=timezone.utc)  # next day: budget resets
    assert usage.get("av") == 0 and usage.try_consume("av", 2)
    assert usage.period("month") == "2026-09"
    usage.mark_exhausted("av", 25)
    assert usage.get("av") == 25 and not usage.try_consume("av", 25)
    usage.mark_exhausted("av", 5)  # never lowers the count
    assert usage.get("av") == 25


def test_usage_is_atomic_across_threads(db):
    usage = UsageTracker(db)
    results = []

    def worker():
        results.append(usage.try_consume("av", 10))

    threads = [threading.Thread(target=worker) for _ in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(results) == 10 and usage.get("av") == 10


def test_quota_guard_per_minute_window():
    clock = [0.0]
    guard = QuotaGuard("p", per_minute=2, clock=lambda: clock[0])
    assert [guard.try_acquire() for _ in range(3)] == [True, True, False]
    clock[0] = 59
    assert guard.try_acquire() is False
    clock[0] = 60.5
    assert guard.try_acquire() is True
    assert guard.remaining() is None


def test_quota_guard_period_budget_persists_across_instances(db):
    usage = UsageTracker(db)
    first = QuotaGuard("av", period_limit=3, usage=usage)
    assert [first.try_acquire() for _ in range(3)] == [True, True, True]
    second = QuotaGuard("av", period_limit=3, usage=usage)  # e.g. after an app restart
    assert second.try_acquire() is False and second.remaining() == 0


def test_denied_minute_check_does_not_burn_period_budget(db):
    usage = UsageTracker(db)
    guard = QuotaGuard("av", per_minute=1, period_limit=10, usage=usage, clock=lambda: 0.0)
    assert guard.try_acquire() and not guard.try_acquire()
    assert usage.get("av") == 1


def test_mark_exhausted_blocks_and_guard_requires_tracker(db):
    guard = QuotaGuard("av", period_limit=25, usage=UsageTracker(db))
    guard.mark_exhausted()
    assert guard.try_acquire() is False and guard.remaining() == 0
    with pytest.raises(ValueError):
        QuotaGuard("x", period_limit=5)
    QuotaGuard("y", per_minute=1).mark_exhausted()  # no-op without a period limit
