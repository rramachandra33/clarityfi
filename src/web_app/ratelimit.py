"""In-memory sliding-window rate limiting for endpoints that spend money (LLM calls, SerpAPI searches).

Per process and per key (a user id or client address). It is a cost/abuse guard for a single-instance
deployment, not a distributed limiter. Exceeding the limit answers HTTP 429 with a Retry-After header.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable

from fastapi import Request

from src.web_app.errors import ApiError


class SlidingWindowLimiter:
    def __init__(self, max_calls: int, window_seconds: float = 60.0, *, max_keys: int = 10_000,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        self.max_calls, self.window, self.max_keys, self._clock = max_calls, window_seconds, max_keys, clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def peek(self, key: str) -> tuple[bool, int]:
        """(allowed, retry_after) WITHOUT recording a call: used to gate sign-in before counting only the failures."""
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if hits and len(hits) >= self.max_calls:
                return False, max(1, math.ceil(self.window - (now - hits[0])))
            return True, 0

    def hit(self, key: str) -> None:
        """Record one call unconditionally (e.g. one failed sign-in)."""
        now = self._clock()
        with self._lock:
            self._hits.setdefault(key, deque()).append(now)

    def check(self, key: str) -> tuple[bool, int]:
        """(allowed, retry_after_seconds). Records the call only when allowed."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.max_calls:
                return False, max(1, math.ceil(self.window - (now - hits[0])))
            hits.append(now)
            if len(self._hits) > self.max_keys:  # bound memory: forget the keys with no recent activity
                for stale in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    self._hits.pop(stale, None)
            return True, 0


def client_key(request: Request, extra: str = "") -> str:
    host = request.client.host if request.client else "unknown"
    return f"{host}:{extra}" if extra else host


def enforce(limiter: SlidingWindowLimiter, key: str) -> None:
    allowed, retry_after = limiter.check(key)
    if not allowed:
        raise ApiError(429, f"Too many requests. Please wait {retry_after} seconds and try again.", "too_many_requests",
                       headers={"Retry-After": str(retry_after)})
