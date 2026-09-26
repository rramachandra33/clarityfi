"""Langfuse tracing helpers.

Tracing is optional: without keys the helpers are no-ops, so tests and local runs never fail
because of observability. Only redacted (post-Presidio) text should ever be passed to spans.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from src.core.config import settings

logger = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def _mask_for_langfuse(*, data: Any, **_: Any) -> Any:
    """Redact PII from everything sent to Langfuse. Fails closed: if redaction errors, send nothing."""
    try:
        from src.core.pii import get_pii_guard

        return get_pii_guard().redact_structure(data)
    except Exception:
        logger.warning("PII masking failed for a trace payload; dropping it", exc_info=True)
        return "[REDACTED: masking error]"


@lru_cache(maxsize=1)
def get_langfuse():
    """Return the shared Langfuse client, or None when tracing is not configured."""
    if not tracing_enabled():
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        mask=_mask_for_langfuse,
    )


@contextmanager
def trace_span(name: str, *, as_type: str = "span", **attributes: Any) -> Iterator[Any]:
    """Open a Langfuse observation for the enclosed block; yields None when tracing is off."""
    client = get_langfuse()
    if client is None:
        yield None
        return
    with client.start_as_current_observation(name=name, as_type=as_type, **attributes) as span:
        yield span


def langfuse_callbacks() -> list[Any]:
    """LangChain/LangGraph callbacks that stream runs to Langfuse ([] when tracing is off or unavailable)."""
    if get_langfuse() is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:  # missing optional dependency etc.: tracing must never break answering
        logger.warning("Langfuse LangChain handler unavailable; continuing without LLM-level tracing", exc_info=True)
        return []


def _mask_for_langsmith(data: Any) -> Any:
    """LangSmith client `anonymizer`: PII-redact every run input/output before upload. Fails closed."""
    try:
        from src.core.pii import get_pii_guard

        return get_pii_guard().redact_structure(data)
    except Exception:
        logger.warning("PII masking failed for a LangSmith payload; dropping it", exc_info=True)
        return {"redacted": "masking error"}


@lru_cache(maxsize=1)
def get_langsmith_client():
    """The shared, PII-masking LangSmith client, or None when LangSmith is not enabled."""
    from src.core.config import langsmith_enabled

    if not langsmith_enabled():
        return None
    from langsmith import Client

    # auto_batch_tracing=False: post runs one by one. Some keys (e.g. personal access tokens in an org) are rejected
    # by the batch endpoint (/runs/multipart, HTTP 403) but accepted on /runs, so this is the portable setting.
    return Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint, anonymizer=_mask_for_langsmith,
                  auto_batch_tracing=False)


def langsmith_callbacks() -> list[Any]:
    """An explicit LangSmith tracer (masked client) when enabled; [] otherwise. Never raises."""
    try:
        client = get_langsmith_client()
        if client is None:
            return []
        from langchain_core.tracers.langchain import LangChainTracer

        return [LangChainTracer(client=client, project_name=settings.langsmith_project)]
    except Exception:
        logger.warning("LangSmith tracer unavailable; continuing without it", exc_info=True)
        return []


def observability_callbacks() -> list[Any]:
    """Every enabled LLM-run tracer: Langfuse (masked) and, if opted in, LangSmith (masked)."""
    return [*langfuse_callbacks(), *langsmith_callbacks()]


def pseudonym(value: str | None, prefix: str = "") -> str | None:
    """Stable, non-reversible id for tracing: a user id may be an email, so the raw value never leaves."""
    if not value:
        return None
    import hashlib

    return prefix + hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()[:16]


@contextmanager
def trace_context(*, user_id: str | None = None, session_id: str | None = None, tags: list[str] | None = None) -> Iterator[None]:
    """Attach (pseudonymous) user/session ids and tags to every observation created inside the block."""
    if get_langfuse() is None:
        yield
        return
    try:
        from langfuse import propagate_attributes

        manager = propagate_attributes(user_id=pseudonym(user_id, "u_"), session_id=pseudonym(session_id, "s_"), tags=tags or None)
    except Exception:
        logger.warning("Langfuse attribute propagation unavailable", exc_info=True)
        yield
        return
    with manager:
        yield


def flush_traces() -> None:
    client = get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception:  # never let observability break the app
            logger.warning("Langfuse flush failed", exc_info=True)
    try:
        ls = get_langsmith_client()
        if ls is not None:
            ls.flush()
    except Exception:
        logger.warning("LangSmith flush failed", exc_info=True)
