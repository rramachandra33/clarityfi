"""Non-blocking quota guard: per-minute sliding window + persistent per-period budget."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

from src.data.cache import UsageTracker


class QuotaGuard:
    """`try_acquire()` returns False instead of sleeping, so callers can fall back to another provider.

    - per_minute: sliding 60s window kept in memory (resets on restart, which is fine for a 60s window)
    - period_limit: persistent budget (daily/monthly) shared across restarts via UsageTracker
    """

    def __init__(
        self,
        provider: str,
        *,
        per_minute: int | None = None,
        period_limit: int | None = None,
        period_kind: str = "day",
        usage: UsageTracker | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if period_limit is not None and usage is None:
            raise ValueError("period_limit requires a UsageTracker")
        self.provider = provider
        self.per_minute = per_minute
        self.period_limit = period_limit
        self.period_kind = period_kind
        self.usage = usage
        self._clock = clock
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            now = self._clock()
            if self.per_minute is not None:
                while self._calls and now - self._calls[0] >= 60:
                    self._calls.popleft()
                if len(self._calls) >= self.per_minute:
                    return False
            if self.period_limit is not None and not self.usage.try_consume(  # type: ignore[union-attr]
                self.provider, self.period_limit, self.period_kind
            ):
                return False
            if self.per_minute is not None:
                self._calls.append(now)
            return True

    def mark_exhausted(self) -> None:
        if self.period_limit is not None and self.usage is not None:
            self.usage.mark_exhausted(self.provider, self.period_limit, self.period_kind)

    def remaining(self) -> int | None:
        if self.period_limit is None or self.usage is None:
            return None
        return max(0, self.period_limit - self.usage.get(self.provider, self.period_kind))
