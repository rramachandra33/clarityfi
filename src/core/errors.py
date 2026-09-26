"""Application exception hierarchy. Messages are safe to show to users (no secrets)."""
from __future__ import annotations


class ClarityError(Exception):
    """Base class for all expected application errors."""

    user_message = "Something went wrong. Please try again."
    code = "error"  # stable machine-readable identifier (used by the API and AgentResult.error_code)

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.user_message)


class ConfigError(ClarityError):
    user_message = "The application is not configured correctly."
    code = "config_error"


class LLMError(ClarityError):
    user_message = "The language model is unavailable right now."
    code = "llm_unavailable"


class AudioError(ClarityError):
    user_message = "Speech services are unavailable right now. Your browser's built-in voice can be used instead."
    code = "audio_unavailable"


class RateLimitedError(LLMError):
    user_message = "The service is busy or over its usage limit. Please try again shortly."
    code = "rate_limited"


class RetrievalError(ClarityError):
    user_message = "The knowledge base could not be searched."
    code = "retrieval_error"


class ProviderError(ClarityError):
    """An external data provider (market data, news, search) failed."""

    user_message = "A data provider is unavailable; showing cached or sample data where possible."
    code = "provider_error"


class GuardrailError(ClarityError):
    """Input or output rejected by validation (length, PII policy, injection heuristics)."""

    user_message = "That request could not be processed."
    code = "guardrail_rejected"


class SymbolNotFoundError(ProviderError):
    """The symbol is unknown to a provider (or malformed)."""

    user_message = "That ticker symbol was not found."
    code = "symbol_not_found"


class ProviderUnavailableError(ProviderError):
    """Network failure, timeout, 5xx, or quota/rate limit: a fallback provider may still work."""

    user_message = "A market-data provider is temporarily unavailable."
    code = "provider_unavailable"


class PrivacyFilterError(ClarityError):
    """The PII filter failed, so nothing was sent onward (fail closed)."""

    user_message = "The privacy filter is unavailable, so your request was not sent."
    code = "privacy_unavailable"


class KnowledgeBaseEmptyError(RetrievalError):
    user_message = "The knowledge base is empty. Upload or ingest a JSON file first."
    code = "knowledge_base_empty"


class EmailError(ClarityError):
    """The email provider failed. Callers that must never reveal account existence should catch and log this,
    not surface it — a password-reset request always answers the same way whether or not the send succeeded."""

    user_message = "The email service is unavailable right now."
    code = "email_unavailable"
