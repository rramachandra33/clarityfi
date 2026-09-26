"""Exponential-backoff retry for transient failures."""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Raise (or wrap with) this inside `fn` to signal a transient failure worth retrying."""


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.25,
    retry_on: tuple[type[BaseException], ...] = (RetryableError,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, retrying on `retry_on` with exponential backoff (base * 2^n, capped, +/- jitter).

    Non-matching exceptions propagate immediately. After the last attempt the final error is raised.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt == attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 1 + random.uniform(-jitter, jitter)
            logger.info("Transient failure (%s); retry %d/%d in %.2fs", type(exc).__name__, attempt, attempts - 1, delay)
            sleep(max(0.0, delay))
    raise AssertionError("unreachable")  # pragma: no cover
