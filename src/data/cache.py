"""SQLite-backed cache and API-usage tracking.

Persisting both means a restart does not re-spend a free-tier quota (Alpha Vantage: ~25 calls/day).
Only public market/news data is cached here, never user text.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

from src.data.db import Database


class PersistentCache:
    """Key/value JSON cache with TTL and 'stale' reads (last-good value regardless of age)."""

    def __init__(self, db: Database, *, clock: Callable[[], float] = time.time) -> None:
        self.db = db
        self._clock = clock

    def set(self, key: str, value: Any) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO kv_cache (key, value, stored_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, stored_at = excluded.stored_at",
                (key, json.dumps(value, default=str), self._clock()),
            )

    def _row(self, key: str):
        with self.db.connect() as conn:
            return conn.execute("SELECT value, stored_at FROM kv_cache WHERE key = ?", (key,)).fetchone()

    def get(self, key: str, ttl_seconds: float) -> Any | None:
        """Value if younger than ttl_seconds, else None."""
        row = self._row(key)
        if row and self._clock() - row["stored_at"] <= ttl_seconds:
            return json.loads(row["value"])
        return None

    def get_with_age(self, key: str, ttl_seconds: float) -> tuple[Any, float] | None:
        """(value, age_seconds) if younger than ttl_seconds, else None."""
        row = self._row(key)
        if row:
            age = self._clock() - row["stored_at"]
            if age <= ttl_seconds:
                return json.loads(row["value"]), age
        return None

    def get_stale(self, key: str, max_age_seconds: float) -> tuple[Any, float] | None:
        """(value, age_seconds) for the last-good value if not older than max_age_seconds."""
        row = self._row(key)
        if not row:
            return None
        age = self._clock() - row["stored_at"]
        return (json.loads(row["value"]), age) if age <= max_age_seconds else None

    def delete_prefix(self, prefix: str) -> int:
        with self.db.connect() as conn:
            return conn.execute("DELETE FROM kv_cache WHERE key LIKE ?", (prefix.replace("%", "") + "%",)).rowcount


class UsageTracker:
    """Counts requests per provider per period (day or month) in the app database."""

    def __init__(self, db: Database, *, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.db = db
        self._now = now

    def period(self, kind: str) -> str:
        n = self._now()
        return n.strftime("%Y-%m-%d") if kind == "day" else n.strftime("%Y-%m")

    def get(self, provider: str, kind: str = "day") -> int:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT count FROM api_usage WHERE provider = ? AND period = ?", (provider, self.period(kind))
            ).fetchone()
        return int(row["count"]) if row else 0

    def try_consume(self, provider: str, limit: int, kind: str = "day") -> bool:
        """Atomically add one request if under `limit`; returns False (without counting) otherwise."""
        period = self.period(kind)
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT OR IGNORE INTO api_usage (provider, period, count) VALUES (?, ?, 0)", (provider, period))
            cur = conn.execute(
                "UPDATE api_usage SET count = count + 1 WHERE provider = ? AND period = ? AND count < ?",
                (provider, period, limit),
            )
            return cur.rowcount == 1

    def mark_exhausted(self, provider: str, limit: int, kind: str = "day") -> None:
        """The provider told us the quota is gone: stop calling it for the rest of the period."""
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO api_usage (provider, period, count) VALUES (?, ?, ?) "
                "ON CONFLICT(provider, period) DO UPDATE SET count = MAX(count, excluded.count)",
                (provider, self.period(kind), limit),
            )
