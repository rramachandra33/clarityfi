"""Structured logging with request/session context and secret masking.

Every record carries the current request_id / session_id / user_id (from contextvars) and any
API-key-shaped strings are masked, so keys can never leak into logs.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Iterator

_CTX_FIELDS = ("request_id", "session_id", "user_id")
_context: ContextVar[dict[str, str]] = ContextVar("log_context", default={})

_SECRET_PATTERNS = [
    re.compile(r"sk-(?:proj-|lf-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"pk-lf-[A-Za-z0-9\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token)(\s*[=:]\s*)[^\s,;'\"]{8,}"),
]


def mask_secrets(text: str) -> str:
    text = _SECRET_PATTERNS[0].sub("sk-***", text)
    text = _SECRET_PATTERNS[1].sub("pk-lf-***", text)
    return _SECRET_PATTERNS[2].sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)


@contextmanager
def log_context(**fields: str) -> Iterator[None]:
    """Bind request_id/session_id/user_id for everything logged inside the block."""
    token = _context.set({**_context.get(), **{k: str(v) for k, v in fields.items() if v is not None}})
    try:
        yield
    finally:
        _context.reset(token)


_UNMASKED_LOGGERS = frozenset({"src.core.email"})
"""Loggers exempt from mask_secrets. src.core.email is the console-mode dev email sender: its whole job
is to print a password-reset link (containing "token=...") where a developer can read it in place of a
real inbox, so masking would silently break local testing of the reset flow."""


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _context.get()
        for name in _CTX_FIELDS:
            setattr(record, name, ctx.get(name))
        record.msg = record.getMessage() if record.name in _UNMASKED_LOGGERS else mask_secrets(record.getMessage())
        record.args = ()
        return True


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"  # YYYY-MM-DD HH24:MI:SS


class JsonFormatter(logging.Formatter):
    """Every line starts with a plain `YYYY-MM-DD HH:MM:SS` timestamp (matching the text formatter and
    uvicorn's own lines), followed by a JSON object with everything else. The timestamp is deliberately
    NOT duplicated inside the JSON: it is the line's one leading, always-in-the-same-place field."""

    def format(self, record: logging.LogRecord) -> str:
        # Local time (not UTC): matches the plain-text formatter and uvicorn's own %(asctime)s, both of
        # which use the system's local time zone by default (logging.Formatter.converter = time.localtime).
        timestamp = datetime.fromtimestamp(record.created).strftime(TIMESTAMP_FORMAT)
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in _CTX_FIELDS:
            value = getattr(record, name, None)
            if value:
                payload[name] = value
        if record.exc_info:
            payload["exception"] = mask_secrets(self.formatException(record.exc_info))
        return f"{timestamp} {json.dumps(payload, ensure_ascii=False)}"


_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure the root logger once (safe to call repeatedly)."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        if getattr(handler, "_clarity", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler._clarity = True  # type: ignore[attr-defined]
    handler.addFilter(ContextFilter())
    handler.setFormatter(JsonFormatter() if json_output else logging.Formatter(_TEXT_FORMAT, datefmt=TIMESTAMP_FORMAT))
    root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
