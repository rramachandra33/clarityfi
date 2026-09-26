from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=3, max_length=120, pattern=r"^[A-Za-z0-9._@\- ]+$")
    conversation_id: str = Field(default="default", min_length=1, max_length=120)
    question: str = Field(min_length=2, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    category: str | None = Field(default=None, max_length=60, pattern=r"^[A-Za-z0-9_ \-]+$",
                                 description="Optional topic filter: a group (e.g. 'tax') or a raw KB category")
    input_mode: Literal["text", "voice"] = "text"
    response_mode: Literal["text", "voice"] = "text"


class SourceResponse(BaseModel):
    index: int
    title: str
    category: str | None = None
    source: str | None = None
    url: str | None = None
    excerpt: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    input_mode: Literal["text", "voice"]
    response_mode: Literal["text", "voice"]
    sources: list[SourceResponse]
    pii_redactions: dict[str, int] = Field(default_factory=dict, description="Entity counts redacted from the question")
    cache_hit: Literal["exact", "semantic"] | None = Field(default=None, description="Set when served from the answer cache")
    data_status: Literal["live", "cached", "stale", "sample"] = "live"
    notice: str | None = Field(default=None, description="Plain-language disclosure; always set when cache_hit is set")
    cached_question: str | None = Field(default=None, description="For semantic hits: the earlier question that was matched")
    agents_used: list[str] = Field(default_factory=list, description="Which agents produced this answer")
    intents: list[str] = Field(default_factory=list, description="What the router decided the question needs")
    router_mode: str | None = Field(default=None, description="heuristic | llm | llm_fallback_heuristic")
    advice_redirect: bool = Field(default=False, description="True when the question asked what to buy/sell and got the education-only note")


class HealthResponse(BaseModel):
    status: str
    openai_ready: bool
    knowledge_chunks: int
    collection: str
    chat_model: str
    embedding_model: str
    glossary_terms: int


class GlossaryItemResponse(BaseModel):
    term: str
    definition: str


class GlossaryListResponse(BaseModel):
    initialized: bool
    count: int
    items: list[GlossaryItemResponse]
    initialization_error: str | None = None


class GlossaryInitializeResponse(BaseModel):
    initialized: bool
    count: int
