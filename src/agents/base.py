"""BaseAgent: the contract every ClarityFI agent implements.

An agent is a thin unit: it takes an `AgentRequest`, uses services (retrieval, market data, calculators,
an LLM) and returns an `AgentResult`. The base class gives every agent the same behaviour for free:

- error containment  : expected errors (ClarityError) and unexpected ones both become an `AgentResult`
                       with `error`/`error_code`; nothing internal is ever shown to the user
- observability      : a Langfuse span per run (pseudonymous user/session ids) and structured logs
- timing             : `data["latency_ms"]` on every result
- LangGraph readiness: `as_node()` returns an async `state -> {"agent_results": [result]}` function, so the
                       workflow (Step 11) can register any agent as a graph node without adapters

Subclasses implement `_execute`. They must attach data-freshness notices (cached/stale/sample) to
`AgentResult.notice` whenever data is not live: see `src/core/freshness.py`.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from src.core.errors import ClarityError
from src.core.models import AgentName, AgentResult
from src.core.tracing import pseudonym, trace_context, trace_span
from src.utils.logging import log_context

logger = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGE = "Something went wrong while answering. Please try again."


class _BackgroundLoop:
    """One long-lived event loop on a daemon thread, used by the synchronous `run()` helpers.

    `asyncio.run()` creates and CLOSES a loop per call, but LangChain/httpx cache async clients bound to
    the first loop, so a second `asyncio.run()` fails with 'Event loop is closed'. A single persistent
    loop keeps those clients valid (FastAPI already provides one for the API path).
    """

    _loop: asyncio.AbstractEventLoop | None = None
    _lock = threading.Lock()

    @classmethod
    def _ensure(cls) -> asyncio.AbstractEventLoop:
        with cls._lock:
            if cls._loop is None or not cls._loop.is_running():
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="agent-loop", daemon=True).start()
                cls._loop = loop
            return cls._loop

    @classmethod
    def run(cls, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run_coroutine_threadsafe(coro, cls._ensure()).result()
        coro.close()
        raise RuntimeError("Agent.run() cannot be used inside a running event loop; use `await agent.arun(...)`.")


@dataclass
class AgentRequest:
    """Everything an agent may need. Text fields must already be validated and PII-redacted."""

    question: str
    user_id: str = "anonymous"
    session_id: str = "default"
    category: str | None = None
    top_k: int = 3
    history: list[dict[str, str]] = field(default_factory=list)  # [{"role": "user"|"assistant", "content": ...}]
    cacheable: bool = True
    context: dict[str, Any] = field(default_factory=dict)        # extras for other agents (portfolio, symbols, ...)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "AgentRequest":
        """Build a request from a LangGraph state mapping."""
        return cls(
            question=str(state["question"]),
            user_id=str(state.get("user_id") or "anonymous"),
            session_id=str(state.get("session_id") or "default"),
            category=state.get("category"),
            top_k=int(state.get("top_k") or 3),
            history=list(state.get("history") or []),
            cacheable=bool(state.get("cacheable", True)),
            context=dict(state.get("context") or {}),
        )


class BaseAgent(ABC):
    name: AgentName
    description: str = ""

    @abstractmethod
    async def _execute(self, request: AgentRequest) -> AgentResult:
        """Do the agent's work. Raise ClarityError subclasses for expected failures."""

    async def arun(self, request: AgentRequest) -> AgentResult:
        started = time.perf_counter()
        agent = self.name.value
        with log_context(session_id=pseudonym(request.session_id, "s_"), user_id=pseudonym(request.user_id, "u_")):
            with trace_context(user_id=request.user_id, session_id=request.session_id, tags=[f"agent:{agent}"]):
                logger.info("Entering %s agent", agent)
                with trace_span(f"agent.{agent}", as_type="agent",
                                input={"question": request.question, "category": request.category}) as span:
                    try:
                        result = await self._execute(request)
                    except ClarityError as exc:
                        logger.warning("%s agent failed: %s [%s]", agent, type(exc).__name__, exc.code)
                        result = AgentResult(agent=self.name, error=str(exc), error_code=exc.code)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception("Unexpected error in %s agent", agent)
                        result = AgentResult(agent=self.name, error=INTERNAL_ERROR_MESSAGE, error_code="internal_error")
                    result.agent = self.name
                    result.data["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
                    if span is not None:
                        span.update(
                            output={"content": result.content, "error_code": result.error_code},
                            metadata={"cache_hit": result.cache_hit, "data_status": result.data_status.value,
                                      "citations": len(result.citations), "latency_ms": result.data["latency_ms"]},
                        )
        logger.info("Leaving %s agent after %.0f ms (error=%s, cache=%s)", agent, result.data["latency_ms"],
                    result.error_code, result.cache_hit)
        return result

    def run(self, request: AgentRequest) -> AgentResult:
        """Synchronous convenience wrapper (scripts, tests). Do not call from inside a running event loop."""
        return _BackgroundLoop.run(self.arun(request))

    def as_node(self) -> Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]:
        """LangGraph node: reads request fields from state, appends its result to `agent_results`."""

        async def node(state: Mapping[str, Any]) -> dict[str, Any]:
            result = await self.arun(AgentRequest.from_state(state))
            return {"agent_results": [result]}

        node.__name__ = f"{self.name.value}_node"
        return node
