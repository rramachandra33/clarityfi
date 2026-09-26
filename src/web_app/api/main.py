from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import PROJECT_ROOT, settings
from src.data.glossary_store import GlossaryAlreadyInitializedError, GlossaryStore, load_glossary_file
from src.rag.loaders import chunk_documents, load_json_documents
from src.core.openai_client import OpenAIClient, OpenAIError
from src.agents.finance_qa import FinanceQAAgent
from src.agents.goal_planning import GoalPlanningAgent
from src.agents.market import MarketAgent
from src.agents.news import NewsAgent
from src.agents.portfolio import PortfolioAgent
from src.agents.tax import TaxEducationAgent
from src.data.market_client import get_market_service
from src.workflow.graph import ChatWorkflow, WorkflowRequest
from src.workflow.memory import SqliteConversationMemory
from src.workflow.router import Router
from src.core.audio import AudioService
from src.core.pii import redact_text
from src.web_app.api.routes_audio import build_audio_router
from src.web_app.api.routes_auth import build_auth_router
from src.web_app.auth_service import AuthService
from src.web_app.security import principal_of, require_admin, require_login
from src.web_app.api.routes_conversations import build_conversations_router
from src.web_app.api.routes_goals import build_goals_router
from src.web_app.api.routes_market import build_market_router
from src.web_app.api.routes_news import build_news_router
from src.web_app.api.routes_portfolio import build_portfolio_router
from src.web_app.api.services import ApiServices
from src.web_app.errors import ApiError, register_error_handlers, status_for
from src.web_app.ratelimit import enforce
from src.data.news_client import get_news_service
from src.web_app.schemas import (
    ChatRequest,
    ChatResponse,
    GlossaryInitializeResponse,
    GlossaryItemResponse,
    GlossaryListResponse,
    HealthResponse,
    SourceResponse,
)
from src.rag.knowledge_base import build_knowledge_base
from src.rag.embedding import CachingEmbedder
from src.rag.qa_cache import QACache
from src.core.config import get_config
from src.data.db import Database
from src.data.seed import seed as seed_synthetic_users
from src.utils.logging import log_context, setup_logging
from src.web_app.api.routes_users import build_users_router

setup_logging(get_config().logging.level, get_config().logging.json_output)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the privacy filter's language model before the first request, so the first user is not the one who waits."""
    try:
        await asyncio.to_thread(redact_text, "warm up")
    except Exception:  # never block startup; requests still fail closed if the filter is really broken
        logger.warning("PII filter warm-up failed; it will be retried on the first request")
    yield


app = FastAPI(
    lifespan=lifespan,
    dependencies=[Depends(require_login)],  # every route needs a signed-in user unless it is in security.PUBLIC_PATHS
    title="Clarity Local Finance RAG API",
    version="1.0.0",
    description="FastAPI, FAISS, and OpenAI backend for financial education Q&A.",
)

app_db = Database(settings.app_db_path)
app_db.init_schema()
seed_synthetic_users(app_db)  # idempotent: only inserts missing synthetic users
app.state.auth = AuthService(app_db)
app.state.auth.ensure_demo_passwords()  # demo users sign in with the documented demo password


def get_db() -> Database:
    return app_db


app.include_router(build_auth_router())
app.include_router(build_users_router(get_db))
register_error_handlers(app)


@app.middleware("http")
async def request_context(request, call_next):
    """Tag every log line with a request id (echoed back as X-Request-ID)."""
    import uuid

    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    with log_context(request_id=request_id):
        response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

openai_client = OpenAIClient(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    chat_model=settings.openai_chat_model,
    embed_model=settings.openai_embed_model,
)
embedder = CachingEmbedder(openai_client)
store = build_knowledge_base(embedder)  # Qdrant hybrid primary, FAISS fallback (config.yaml)
_cache_cfg = get_config().cache
qa_cache = (QACache(app_db, exact_ttl=_cache_cfg.qa_exact_ttl_seconds, semantic_ttl=_cache_cfg.qa_semantic_ttl_seconds,
                    threshold=_cache_cfg.qa_semantic_threshold, max_entries=_cache_cfg.qa_max_entries)
            if _cache_cfg.qa_enabled else None)
if qa_cache is not None:
    store.on_change.append(qa_cache.bump_version)  # new/changed knowledge invalidates every cached answer
finance_qa = FinanceQAAgent(store, cache=qa_cache, embed=lambda text: embedder.embed([text])[0])
market = get_market_service()
agents = {
    "finance_qa": finance_qa,
    "portfolio": PortfolioAgent(app_db, market),
    "market": MarketAgent(market),
    "goal_planning": GoalPlanningAgent(app_db),
    "news": NewsAgent(),
    "tax": TaxEducationAgent(store, cache=qa_cache, embed=lambda text: embedder.embed([text])[0]),
}
memory = SqliteConversationMemory(app_db)  # chat history persisted in SQLite (redacted text only)
workflow = ChatWorkflow(agents, Router(use_llm=get_config().workflow.router_use_llm), memory)

glossary_store = GlossaryStore(settings.glossary_db_path)
glossary_seed_path = PROJECT_ROOT / "data/financial_glossary.json"
glossary_initialization_error: str | None = None
if glossary_store.count() == 0 and glossary_seed_path.exists():
    try:
        glossary_store.initialize_from_file(glossary_seed_path)
    except (ValueError, FileNotFoundError, GlossaryAlreadyInitializedError) as exc:
        glossary_initialization_error = str(exc)

api_services = ApiServices(db=app_db, market=market, news=get_news_service(), portfolio_agent=agents["portfolio"],
                           goal_agent=agents["goal_planning"], news_agent=agents["news"], memory=memory, audio=AudioService())


def get_services() -> ApiServices:
    return api_services


for _build in (build_audio_router, build_portfolio_router, build_market_router, build_goals_router, build_news_router, build_conversations_router):
    app.include_router(_build(get_services))

@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        count = store.count()
    except Exception:
        count = 0
    ready = openai_client.is_available()
    return HealthResponse(
        status="ready" if ready and count > 0 else "needs_setup",
        openai_ready=ready,
        knowledge_chunks=count,
        collection=settings.faiss_collection,
        chat_model=settings.openai_chat_model,
        embedding_model=settings.openai_embed_model,
        glossary_terms=glossary_store.count(),
    )


@app.post("/api/ingest/file", dependencies=[Depends(require_admin)])
async def ingest_file(
    file: UploadFile = File(...),
    reset: bool = Form(default=False),
) -> dict[str, int | str]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".json", ".jsonl", ".ndjson"}:
        raise HTTPException(status_code=400, detail="Upload a .json, .jsonl, or .ndjson file")

    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temp_path = Path(handle.name)
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit")
                handle.write(chunk)
        documents = load_json_documents(temp_path)
        for document in documents:
            document.metadata["source"] = file.filename or "uploaded.json"
        chunks = chunk_documents(documents)
        ingested = store.ingest(chunks, reset=reset)
        return {
            "file": file.filename or "uploaded.json",
            "documents": len(documents),
            "chunks_ingested": ingested,
            "total_chunks": store.count(),
        }
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await file.close()
        if temp_path:
            temp_path.unlink(missing_ok=True)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http: Request) -> ChatResponse:
    """One entry point for every question: the workflow validates, redacts, routes to one or more agents
    (in parallel), merges their answers, and redacts the result again before it leaves."""
    caller = principal_of(http)
    if not caller.owns(request.user_id):
        raise ApiError(403, "You can only chat as yourself.", "forbidden")
    enforce(get_services().limiters["chat"], request.user_id.strip().lower())  # per-user cost guard: each turn can call several LLMs
    result = await workflow.ainvoke(WorkflowRequest(
        question=request.question, user_id=request.user_id, conversation_id=request.conversation_id,
        category=request.category, top_k=request.top_k or settings.rag_top_k))
    if result.error:
        raise HTTPException(status_code=status_for(result.error_code), detail=result.error)
    return ChatResponse(
        answer=result.content,
        conversation_id=request.conversation_id,
        input_mode=request.input_mode,
        response_mode=request.response_mode,
        sources=[SourceResponse(index=c.index, title=c.title, category=c.category, source=c.source, url=c.url,
                                excerpt=c.excerpt or "", score=c.score or 0.0) for c in result.citations],
        pii_redactions=result.pii_redactions,
        cache_hit=result.cache_hit,
        data_status=result.data_status.value,
        notice=result.notice,
        cached_question=result.cached_question,
        agents_used=result.agents_used,
        intents=result.intents,
        router_mode=result.router_mode,
        advice_redirect=result.advice_redirect,
    )


@app.get("/api/stats")
def stats() -> dict:
    return {
        "collection": settings.faiss_collection,
        "knowledge_chunks": store.count(),
        "glossary_terms": glossary_store.count(),
        "qa_cache": qa_cache.stats() if qa_cache else None,
    }


@app.get("/api/glossary", response_model=GlossaryListResponse)
def glossary(
    q: str = Query(default="", max_length=200),
    limit: int = Query(default=1000, ge=1, le=5000),
) -> GlossaryListResponse:
    count = glossary_store.count()
    entries = glossary_store.list(query=q, limit=limit) if count else []
    return GlossaryListResponse(
        initialized=count > 0,
        count=count,
        items=[GlossaryItemResponse(term=entry.term, definition=entry.definition) for entry in entries],
        initialization_error=glossary_initialization_error,
    )


@app.post("/api/glossary/initialize", response_model=GlossaryInitializeResponse, dependencies=[Depends(require_admin)])
async def initialize_glossary(file: UploadFile = File(...)) -> GlossaryInitializeResponse:
    global glossary_initialization_error
    existing = glossary_store.count()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Glossary is already initialized with {existing} terms and is read-only",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".json", ".jsonl", ".ndjson"}:
        raise HTTPException(status_code=400, detail="Upload financial_glossary.json, .jsonl, or .ndjson")

    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temp_path = Path(handle.name)
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit")
                handle.write(chunk)
        entries = load_glossary_file(temp_path)
        count = glossary_store.initialize(entries)
        glossary_initialization_error = None
        return GlossaryInitializeResponse(initialized=True, count=count)
    except HTTPException:
        raise
    except GlossaryAlreadyInitializedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        glossary_initialization_error = str(exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
        if temp_path:
            temp_path.unlink(missing_ok=True)
