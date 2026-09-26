"""The chat workflow (LangGraph): guard -> router -> agents (parallel) -> synthesizer -> output guard.

    question ──► guard ──► router ──┬─► agent A ─┐
                  │                 ├─► agent B ─┼─► synthesize ──► output_guard ──► response
                  └──(rejected)─────┴────────────┘

- guard        : validate, check for harmful content (self-harm / violence, short-circuits straight to a
                 deterministic safety response - see `core/safety.py`), flag prompt-injection patterns,
                 PII-redact (fail closed). Only the REDACTED text continues; the raw question never
                 reaches an agent, the database, a cache, or a trace.
- router       : rules for clear cases, LLM only for unsure ones (see router.py). One question may need
                 several agents; they run IN PARALLEL (LangGraph `Send`), each with its own timeout.
- synthesize   : merges agent answers into one reply: sections + ONE data-freshness note + ONE disclaimer,
                 partial failures explained, education-only note for "should I buy...?" questions.
- output_guard : PII-redacts the final text (fail closed) and guarantees the disclaimer, except for the
                 self-harm safety response, which is static app text and skips both (see `skip_output_guard`).
- memory       : conversation history is loaded before the graph and the exchange persisted after it
                 (redacted text only; SQLite-backed in production). Refusals, including harmful-content
                 short-circuits, are never persisted.

Everything is traced to Langfuse as one `workflow.chat` trace: the router decision and each agent (with its
LLM generation) appear as nested observations. The trace input is the REDACTED question.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.base import AgentRequest, BaseAgent
from src.core.config import get_config
from src.core.disclaimers import ADVICE_REDIRECT, PLAIN_DISCLAIMER, ensure_educational_disclaimer, with_data_notices
from src.core.errors import GuardrailError
from src.core.freshness import DataStatus, combine_status
from src.core.models import AgentName, AgentResult, Citation
from src.core.pii import redact_output, redact_text
from src.core.safety import HARMFUL_CONTENT_RESPONSE, SELF_HARM_RESPONSE, detect_harmful_content
from src.core.tracing import pseudonym, trace_context, trace_span
from src.utils.logging import log_context
from src.utils.validators import looks_like_prompt_injection, validate_user_input
from src.workflow.router import Router, focus_question

logger = logging.getLogger(__name__)

SECTION_TITLES = {
    "portfolio": "Your portfolio", "goal_planning": "Your goal", "market": "Market", "news": "News", "tax": "Taxes",
    "finance_qa": "Background",
}
SECTION_ORDER = {name: i for i, name in enumerate(["portfolio", "goal_planning", "market", "news", "tax", "finance_qa"])}
OUT_OF_SCOPE_MESSAGE = (
    "I'm here to help with personal finance and investing education: explaining concepts, analyzing a portfolio, "
    "looking up prices and trends, planning savings goals, and tax basics. I can't help with that topic. "
    "Educational only."
)
NEWS_UNAVAILABLE_NOTE = "Live news summaries aren't available yet, so I can't say what the headlines are saying."
PRIVACY_UNAVAILABLE = "The privacy filter is unavailable, so your message was not processed."
OUTPUT_PRIVACY_UNAVAILABLE = "The privacy filter is unavailable, so the answer was withheld."
_DATA_NOTE = re.compile(r"\n*Data note:\n((?:- .*(?:\n|$))+)\s*$")


# ------------------------------------------------------------------------------------- data types
class WorkflowState(TypedDict, total=False):
    # input
    question: str
    user_id: str
    session_id: str
    category: str | None
    top_k: int
    history: list[dict[str, str]]
    # after guard
    text: str
    cacheable: bool
    pii_counts: dict[str, int]
    injection_suspected: bool
    harmful_category: str | None
    error: dict[str, str] | None
    # after router
    intents: list[str]
    router_mode: str
    router_reason: str
    advice_request: bool
    # parallel agent results (reducer: concatenate)
    agent_results: Annotated[list[AgentResult], add]
    # after synthesize / output_guard
    final: dict[str, Any]


@dataclass
class WorkflowRequest:
    question: str
    user_id: str
    conversation_id: str = "default"
    category: str | None = None
    top_k: int = 3


@dataclass
class WorkflowResponse:
    content: str = ""
    citations: list[Citation] = field(default_factory=list)
    notice: str | None = None
    data_status: DataStatus = DataStatus.live
    cache_hit: str | None = None
    cached_question: str | None = None
    agents_used: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    router_mode: str | None = None
    advice_redirect: bool = False
    pii_redactions: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    error_code: str | None = None
    refusal: bool = False
    agent_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


# ------------------------------------------------------------------------------- text utilities
def split_data_note(content: str) -> tuple[str, list[str]]:
    """Separate a trailing 'Data note:' block (added by data agents) from the body text."""
    match = _DATA_NOTE.search(content)
    if not match:
        return content, []
    notes = [line[2:].strip() for line in match.group(1).splitlines() if line.startswith("- ")]
    return content[: match.start()].rstrip(), notes


def strip_closing_disclaimer(text: str) -> str:
    """Remove a trailing 'Educational only...' sentence so merged sections carry one disclaimer at the end."""
    out = text.rstrip()
    if out.endswith(PLAIN_DISCLAIMER):
        return out[: -len(PLAIN_DISCLAIMER)].rstrip()
    return re.sub(r"\s*Educational only\.?$", "", out).rstrip()


# ------------------------------------------------------------------------------------ the workflow
class ChatWorkflow:
    def __init__(self, agents: dict[str, BaseAgent], router: Router, memory, *,
                 agent_timeout: float | None = None, max_input_chars: int | None = None) -> None:
        cfg = get_config()
        self.agents, self.router, self.memory = agents, router, memory
        self.agent_timeout = cfg.workflow.agent_timeout_seconds if agent_timeout is None else agent_timeout
        self.max_input_chars = cfg.guardrails.max_input_chars if max_input_chars is None else max_input_chars
        self.graph = self._build()

    # -- nodes ----------------------------------------------------------------------------------
    async def _guard(self, state: WorkflowState) -> dict[str, Any]:
        try:
            text = validate_user_input(state["question"], self.max_input_chars)
        except GuardrailError as exc:
            return {"error": {"code": exc.code, "message": str(exc)}}
        harm = detect_harmful_content(text)
        if harm:
            logger.warning("Harmful-content pattern detected (%s); request short-circuited before routing", harm)
            return {"harmful_category": harm}
        injection = looks_like_prompt_injection(text)
        if injection:
            logger.warning("Possible prompt-injection pattern in chat input")
        try:
            redaction = await asyncio.to_thread(redact_text, text)
        except Exception:
            logger.exception("PII filter failed; request blocked")
            return {"error": {"code": "privacy_unavailable", "message": PRIVACY_UNAVAILABLE}}
        if redaction.redacted:
            logger.info("Redacted PII from chat input: %s", redaction.counts)
        return {"text": redaction.text, "pii_counts": redaction.counts, "cacheable": not redaction.redacted,
                "injection_suspected": injection}

    async def _route(self, state: WorkflowState) -> dict[str, Any]:
        with trace_span("workflow.route", as_type="span", input={"question": state["text"]}) as span:
            decision = await self.router.route(state["text"])
            if span is not None:
                span.update(output={"intents": decision.intents, "mode": decision.mode, "advice_request": decision.advice_request},
                            metadata={"reason": decision.reason, "signals": decision.signals})
        logger.info("Routed to %s (%s)", decision.intents, decision.mode)
        return {"intents": decision.intents, "router_mode": decision.mode, "router_reason": decision.reason,
                "advice_request": decision.advice_request}

    def _plan(self, state: WorkflowState) -> tuple[list[tuple[str, str | None, str]], list[str]]:
        """intents -> [(agent name, category override, intent that caused it)] plus intents with no agent yet."""
        plan: dict[str, tuple[str | None, str]] = {}
        unavailable: list[str] = []
        for intent in state.get("intents", []):
            if intent == "out_of_scope":
                continue
            if intent in self.agents:
                plan.setdefault(intent, (state.get("category"), intent))
            elif intent == "tax" and "finance_qa" in self.agents:
                plan["finance_qa"] = (state.get("category") or "tax", "tax")  # tax questions -> Q&A over tax content
            else:
                unavailable.append(intent)
        return [(name, cat, why) for name, (cat, why) in plan.items()], unavailable

    def _fan_out(self, state: WorkflowState):
        if state.get("error"):
            return "synthesize"
        plan, _unavailable = self._plan(state)
        if not plan:
            return "synthesize"
        # Each agent gets only ITS part of a multi-part question, so it never comments on the parts it cannot see.
        return [Send("agent", {"name": name, "category": category, "text": focus_question(state["text"], why),
                               "user_id": state["user_id"], "session_id": state["session_id"], "top_k": state.get("top_k", 3),
                               "history": state.get("history", []), "cacheable": state.get("cacheable", True)})
                for name, category, why in plan]

    async def _agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = payload["name"]
        agent = self.agents[name]
        request = AgentRequest(question=payload["text"], user_id=payload["user_id"], session_id=payload["session_id"],
                               category=payload["category"], top_k=payload["top_k"], history=payload["history"],
                               cacheable=payload["cacheable"])
        try:
            result = await asyncio.wait_for(agent.arun(request), timeout=self.agent_timeout)
        except asyncio.TimeoutError:
            logger.warning("%s agent timed out after %.0fs", name, self.agent_timeout)
            result = AgentResult(agent=AgentName(name), error="That part took too long to finish.", error_code="agent_timeout")
        return {"agent_results": [result]}

    async def _synthesize(self, state: WorkflowState) -> dict[str, Any]:
        harm = state.get("harmful_category")
        if harm == "self_harm":  # static app text, not derived from user input: skip PII redaction/disclaimer
            return {"final": {"content": SELF_HARM_RESPONSE, "agents_used": [], "refusal": True, "skip_output_guard": True}}
        if harm == "violence":
            return {"final": {"content": HARMFUL_CONTENT_RESPONSE, "agents_used": [], "refusal": True}}
        if state.get("error"):
            err = state["error"]
            return {"final": {"error": err["message"], "error_code": err["code"]}}
        intents = state.get("intents", [])
        results: list[AgentResult] = list(state.get("agent_results", []))
        unavailable = self._plan(state)[1]

        if intents == ["out_of_scope"] or (not results and not unavailable):
            return {"final": {"content": OUT_OF_SCOPE_MESSAGE, "agents_used": [], "refusal": True}}

        results.sort(key=lambda r: SECTION_ORDER.get(r.agent.value, 99))
        good = [r for r in results if r.ok]
        failed = [r for r in results if not r.ok]
        if not good and failed and not unavailable:
            first = failed[0]
            return {"final": {"error": first.error, "error_code": first.error_code,
                              "agents_used": [r.agent.value for r in failed]}}

        sections: list[tuple[str, str]] = []
        notes: list[str] = []
        citations: list[Citation] = []
        offset = 0
        renumber = sum(1 for r in good if r.citations) > 1  # several agents cite: [1] would collide, so renumber
        for r in good:
            body, data_notes = split_data_note(r.content)
            notes.extend(data_notes)
            if renumber and r.citations:
                count = len(r.citations)
                body = re.sub(r"\[(\d{1,2})\]", lambda m: f"[{int(m.group(1)) + offset}]" if 1 <= int(m.group(1)) <= count else m.group(0), body)
                citations.extend(c.model_copy(update={"index": c.index + offset}) for c in r.citations)
                offset += count
            else:
                citations.extend(r.citations)
            sections.append((r.agent.value, strip_closing_disclaimer(body) if len(good) + len(unavailable) > 1 else body))
        for name in unavailable:
            sections.append((name, NEWS_UNAVAILABLE_NOTE if name == "news" else
                             f"The {SECTION_TITLES.get(name, name).lower()} feature isn't available right now."))
        multi = len(sections) > 1
        parts = [f"{SECTION_TITLES.get(name, name)}\n{text}" if multi else text for name, text in sections]
        for r in failed:
            parts.append(f"I couldn't complete the {SECTION_TITLES.get(r.agent.value, r.agent.value).lower()} part: {r.error}")
        content = "\n\n".join(p for p in parts if p.strip())
        if state.get("advice_request"):
            content = f"{ADVICE_REDIRECT}\n\n{content}"
        if multi:
            content = ensure_educational_disclaimer(with_data_notices(content, list(dict.fromkeys(notes))))

        all_notices = list(dict.fromkeys([*notes, *[r.notice for r in good if r.notice]]))
        qa = next((r for r in good if r.cache_hit), None)
        refusal = bool(good) and all(r.data.get("refusal") for r in good) and not unavailable
        return {"final": {
            "content": content, "citations": citations,
            "notice": " ".join(all_notices) or None, "data_status": combine_status([r.data_status for r in good]),
            "cache_hit": qa.cache_hit if qa else None, "cached_question": qa.data.get("cached_question") if qa else None,
            "agents_used": [r.agent.value for r in good], "refusal": refusal, "advice_redirect": bool(state.get("advice_request")),
            "agent_data": {r.agent.value: r.data for r in good},
        }}

    async def _output_guard(self, state: WorkflowState) -> dict[str, Any]:
        final = dict(state.get("final", {}))
        if final.get("error") or not final.get("content") or final.get("skip_output_guard"):
            return {"final": final}
        try:
            final["content"] = (await asyncio.to_thread(redact_output, final["content"])).text
        except Exception:
            logger.exception("PII filter failed on the final answer; response withheld")
            return {"final": {"error": OUTPUT_PRIVACY_UNAVAILABLE, "error_code": "privacy_unavailable"}}
        final["content"] = ensure_educational_disclaimer(final["content"])
        return {"final": final}

    # -- graph ----------------------------------------------------------------------------------
    def _build(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("guard", self._guard)
        graph.add_node("router", self._route)
        graph.add_node("agent", self._agent)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("output_guard", self._output_guard)
        graph.add_edge(START, "guard")
        graph.add_conditional_edges(
            "guard", lambda s: "synthesize" if (s.get("error") or s.get("harmful_category")) else "router",
            ["router", "synthesize"])
        graph.add_conditional_edges("router", self._fan_out, ["agent", "synthesize"])
        graph.add_edge("agent", "synthesize")
        graph.add_edge("synthesize", "output_guard")
        graph.add_edge("output_guard", END)
        return graph.compile()

    # -- entry point ----------------------------------------------------------------------------
    async def ainvoke(self, request: WorkflowRequest) -> WorkflowResponse:
        started = time.perf_counter()
        key = self.memory.key(request.user_id, request.conversation_id)
        history = await asyncio.to_thread(self.memory.history, key)
        with log_context(session_id=pseudonym(request.conversation_id, "s_"), user_id=pseudonym(request.user_id, "u_")):
            with trace_context(user_id=request.user_id, session_id=request.conversation_id, tags=["workflow:chat"]):
                with trace_span("workflow.chat", as_type="chain") as span:
                    state = await self.graph.ainvoke({
                        "question": request.question, "user_id": request.user_id, "session_id": request.conversation_id,
                        "category": request.category, "top_k": request.top_k, "history": history, "agent_results": []})
                    response = self._response(state)
                    response.latency_ms = round((time.perf_counter() - started) * 1000, 1)
                    if span is not None:  # the trace input is the REDACTED question, never the raw one
                        span.update(input={"question": state.get("text", "")},
                                    output={"content": response.content, "error_code": response.error_code},
                                    metadata={"intents": response.intents, "router_mode": response.router_mode,
                                              "agents_used": response.agents_used, "cache_hit": response.cache_hit,
                                              "data_status": response.data_status.value, "latency_ms": response.latency_ms})
        if response.ok and not response.refusal and response.content:
            try:
                await asyncio.to_thread(
                    self.memory.record_exchange, key, state["text"], response.content,
                    agent=",".join(response.agents_used) or None, cache_hit=response.cache_hit,
                    citations=[c.model_dump(mode="json") for c in response.citations] or None)
            except Exception:  # losing a history row must never lose the answer
                logger.exception("Could not persist the conversation turn")
        return response

    @staticmethod
    def _response(state: WorkflowState) -> WorkflowResponse:
        final = state.get("final", {})
        return WorkflowResponse(
            content=final.get("content", ""), citations=final.get("citations", []), notice=final.get("notice"),
            data_status=final.get("data_status", DataStatus.live), cache_hit=final.get("cache_hit"),
            cached_question=final.get("cached_question"), agents_used=final.get("agents_used", []),
            intents=state.get("intents", []), router_mode=state.get("router_mode"), advice_redirect=final.get("advice_redirect", False),
            pii_redactions=state.get("pii_counts", {}), error=final.get("error"), error_code=final.get("error_code"),
            refusal=final.get("refusal", False), agent_data=final.get("agent_data", {}))
