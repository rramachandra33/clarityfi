from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# src/core/config.py -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def apply_dotenv_overrides(path: Path, prefixes: tuple[str, ...] = ("LANGSMITH_", "ENABLE_LANGSMITH")) -> list[str]:
    """Let .env WIN over an already-set shell variable for the given prefixes; returns the names overridden.

    `load_dotenv` never overrides existing environment variables, and a stale LANGSMITH_API_KEY left in a shell
    profile silently replaced the key in .env (uploads were then rejected with HTTP 403). For LangSmith the
    project's .env is authoritative.
    """
    from dotenv import dotenv_values

    overridden = []
    for name, value in dotenv_values(path).items():
        if value is not None and name.startswith(prefixes) and os.environ.get(name) != value:
            os.environ[name] = value
            overridden.append(name)
    return overridden


apply_dotenv_overrides(PROJECT_ROOT / ".env")


def langsmith_enabled() -> bool:
    return os.getenv("ENABLE_LANGSMITH", "").strip().lower() in {"1", "true", "yes"} and bool(os.getenv("LANGSMITH_API_KEY", "").strip())


def _disable_langsmith_env_tracing() -> None:
    """Turn OFF LangChain's automatic environment-driven LangSmith uploads, always.

    With LANGSMITH_TRACING(_V2)/LANGCHAIN_TRACING_V2 enabled in the shell, LangChain uploads every run
    (prompts, retrieved passages, answers) to LangSmith with no PII masking. We never rely on that path.
    LangSmith is used only when ENABLE_LANGSMITH=true AND a key is set in .env, through an explicit tracer
    whose client redacts PII first (see `src/core/tracing.py`).
    """
    for name in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        os.environ[name] = "false"


_disable_langsmith_env_tracing()


def _path(env_name: str, default: Path) -> Path:
    path = Path(os.getenv(env_name, str(default))).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    glossary_db_path: Path = _path("GLOSSARY_DB_PATH", PROJECT_ROOT / "data/glossary.db")
    faiss_path: Path = _path("FAISS_PATH", PROJECT_ROOT / "data/faiss")
    faiss_collection: str = os.getenv("FAISS_COLLECTION", "finance_education")
    app_db_path: Path = _path("APP_DB_PATH", PROJECT_ROOT / "data/clarityfi.db")
    qdrant_path: Path = _path("QDRANT_PATH", PROJECT_ROOT / "data/qdrant")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "finance_education")
    qdrant_url: str = os.getenv("QDRANT_URL", "")  # empty = embedded local mode
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "3"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_api_key: str = field(default=os.getenv("OPENAI_API_KEY", ""), repr=False)
    openai_embed_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    serpapi_api_key: str = field(default=os.getenv("SERPAPI_API_KEY", ""), repr=False)
    alphavantage_api_key: str = field(default=os.getenv("ALPHAVANTAGE_API_KEY", ""), repr=False)
    langfuse_secret_key: str = field(default=os.getenv("LANGFUSE_SECRET_KEY", ""), repr=False)
    langfuse_public_key: str = field(default=os.getenv("LANGFUSE_PUBLIC_KEY", ""), repr=False)
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    langsmith_api_key: str = field(default=os.getenv("LANGSMITH_API_KEY", ""), repr=False)
    langsmith_endpoint: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "clarityfi")
    admin_user_ids: str = os.getenv("ADMIN_USER_IDS", "")          # comma-separated ids allowed to upload knowledge / glossary files
    demo_login_enabled: bool = os.getenv("DEMO_LOGIN_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
    demo_password: str = field(default=os.getenv("DEMO_PASSWORD", "").strip() or "clarity-demo", repr=False)  # for the 10 synthetic demo users only
    app_base_url: str = os.getenv("APP_BASE_URL", "http://127.0.0.1:5173").rstrip("/")  # used to build links in emails (e.g. password reset)
    email_provider: str = os.getenv("EMAIL_PROVIDER", "console").strip().lower()  # console (dev: logs instead of sending) | ses
    ses_sender_email: str = os.getenv("SES_SENDER_EMAIL", "")   # must be a verified SES identity
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")


settings = Settings()


# ---- config.yaml (non-secret tunables) --------------------------------------------------------
class LLMConfig(BaseModel):
    temperature: float = 0.1
    timeout_seconds: float = 60
    max_retries: int = 2
    max_output_tokens: int = 900


class RagConfig(BaseModel):
    backend: str = "qdrant"          # primary retriever: qdrant (hybrid) | faiss (dense only)
    fallback: str = "faiss"          # used automatically if the primary errors or is empty ("" disables)
    candidates: int = 20             # per-list candidates fed into RRF
    rrf_k: int = 60
    top_k: int = 3
    sourced_bonus: float = 0.08     # ranking-only boost for articles with a real source link (never changes the shown score)
    overfetch: int = 3              # candidates fetched per wanted result so the boost has something to reorder
    score_threshold: float = 0.25
    chunk_chars: int = 1200
    chunk_overlap: int = 150


class CacheConfig(BaseModel):
    qa_enabled: bool = True
    qa_max_entries: int = 500
    qa_exact_ttl_seconds: int = 86400
    qa_semantic_ttl_seconds: int = 86400
    qa_semantic_threshold: float = 0.90
    market_ttl_seconds: int = 1800
    news_ttl_seconds: int = 1800


class MarketConfig(BaseModel):
    providers: list[str] = Field(default_factory=lambda: ["alphavantage", "yfinance", "mock"])
    request_timeout_seconds: float = 10
    max_retries: int = 3
    retry_base_delay_seconds: float = 0.5
    alphavantage_calls_per_minute: int = 5
    alphavantage_daily_limit: int = 25       # free tier
    serpapi_monthly_limit: int = 100         # free tier
    history_days: int = 180
    history_ttl_seconds: int = 21600         # 6 hours
    stale_max_age_seconds: int = 604800      # serve last-good data up to 7 days old when providers are down


class GuardrailConfig(BaseModel):
    max_input_chars: int = 4000
    pii_entities: list[str] = Field(
        default_factory=lambda: ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"]
    )
    pii_score_threshold: float = 0.5
    pii_output_entities: list[str] = Field(
        default_factory=lambda: ["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IBAN_CODE", "US_BANK_NUMBER", "IP_ADDRESS"]
    )
    pii_entity_thresholds: dict[str, float] = Field(default_factory=lambda: {"PHONE_NUMBER": 0.4, "US_BANK_NUMBER": 0.4})
    pii_model: str = "en_core_web_lg"


class WorkflowConfig(BaseModel):
    agent_timeout_seconds: float = 45     # one slow agent must not hold up the whole answer
    router_use_llm: bool = True           # False = rules only (cheaper, less accurate on unusual phrasing)


class ApiConfig(BaseModel):
    chat_per_minute: int = 20          # per user id: each chat turn may cost several LLM calls
    expensive_per_minute: int = 12     # per client: LLM narration / news search endpoints
    read_per_minute: int = 240         # per client: cheap reads (a dashboard polls a few endpoints)


class AudioConfig(BaseModel):
    stt_model: str = "whisper-1"
    tts_model: str = "tts-1"
    voice: str = "alloy"
    max_upload_bytes: int = 5_000_000   # ~5 MB: a few minutes of compressed speech; audio is never stored
    max_tts_chars: int = 1500
    timeout_seconds: float = 45
    allowed_types: list[str] = Field(default_factory=lambda: [
        "audio/webm", "audio/ogg", "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave"])


class AuthConfig(BaseModel):
    session_days: int = 7
    min_password_length: int = 8
    max_password_length: int = 128
    login_attempts: int = 5             # failed attempts per (client, user id) ...
    login_window_seconds: int = 900     # ... within this window before a 429
    ip_login_attempts: int = 30         # failed attempts per client (any user id) in the same window
    cookie_name: str = "clarity_session"
    reset_token_ttl_seconds: int = 3600         # a password-reset link is valid for one hour
    reset_requests_per_window: int = 5          # per client, in login_window_seconds: guards against inbox-flooding / enumeration timing


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_output: bool = Field(default=True, alias="json")

    model_config = {"populate_by_name": True}


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load config.yaml (CONFIG_FILE env overrides the path). Missing file -> defaults."""
    config_path = path or _path("CONFIG_FILE", PROJECT_ROOT / "config.yaml")
    data: dict[str, Any] = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(data)
    if os.getenv("RAG_TOP_K"):
        config.rag.top_k = int(os.environ["RAG_TOP_K"])
    return config


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_app_config()
