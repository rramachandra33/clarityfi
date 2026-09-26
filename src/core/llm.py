"""LangChain OpenAI wrappers (chat + embeddings) with consistent error mapping.

Agents and the LangGraph workflow use these instead of talking to the OpenAI SDK directly.
The legacy httpx client in `openai_client.py` remains for the existing RAG path until Step 9.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.core.config import get_config, settings
from src.core.errors import ConfigError, LLMError, RateLimitedError

logger = logging.getLogger(__name__)


def model_name(model: Any) -> str:
    """Best-effort model name for logging (works for ChatOpenAI, its structured-output wrappers, and test fakes)."""
    return getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__


def get_chat_model(**overrides: Any) -> BaseChatModel:
    """Build the configured ChatOpenAI model. Raises ConfigError if no API key is set."""
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise ConfigError("OPENAI_API_KEY is not set. Add it to .env.")
    cfg = get_config().llm
    params: dict[str, Any] = {
        "model": settings.openai_chat_model,
        "api_key": settings.openai_api_key,
        "base_url": settings.openai_base_url,
        "temperature": cfg.temperature,
        "timeout": cfg.timeout_seconds,
        "max_retries": cfg.max_retries,
        "max_tokens": cfg.max_output_tokens,
    }
    params.update(overrides)
    return ChatOpenAI(**params)


def get_embeddings(**overrides: Any):
    from langchain_openai import OpenAIEmbeddings

    if not settings.openai_api_key:
        raise ConfigError("OPENAI_API_KEY is not set. Add it to .env.")
    params: dict[str, Any] = {
        "model": settings.openai_embed_model,
        "api_key": settings.openai_api_key,
        "base_url": settings.openai_base_url,
    }
    params.update(overrides)
    return OpenAIEmbeddings(**params)


def map_llm_exception(exc: Exception) -> LLMError:
    """Translate provider exceptions into safe, user-presentable errors."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name == "RateLimitError" or status == 429:
        return RateLimitedError()
    if name == "AuthenticationError" or status in (401, 403):
        return LLMError("OpenAI authentication failed. Check OPENAI_API_KEY.")
    if name in {"APITimeoutError", "Timeout", "TimeoutError"}:
        return LLMError("The language model timed out. Please try again.")
    if name == "APIConnectionError":
        return LLMError("Could not reach the language model service.")
    return LLMError()


def invoke_chat(
    messages: Sequence[BaseMessage] | str,
    *,
    model: BaseChatModel | None = None,
    callbacks: list[Any] | None = None,
    purpose: str = "chat completion",
    **kwargs: Any,
) -> str:
    """Run one chat completion and return the text, raising LLMError on failure."""
    llm = model or get_chat_model()
    started = time.perf_counter()
    logger.info("Calling LLM for %s (model=%s)", purpose, model_name(llm))
    try:
        response = llm.invoke(messages, config={"callbacks": callbacks or []}, **kwargs)
    except (ConfigError, LLMError):
        raise
    except Exception as exc:  # provider SDKs raise many types
        logger.warning("LLM call failed: %s", type(exc).__name__)
        raise map_llm_exception(exc) from exc
    logger.info("LLM call for %s succeeded in %.0f ms", purpose, (time.perf_counter() - started) * 1000)
    content = response.content
    return content if isinstance(content, str) else "".join(str(part) for part in content)


async def ainvoke_chat(
    messages: Sequence[BaseMessage] | str,
    *,
    model: BaseChatModel | None = None,
    callbacks: list[Any] | None = None,
    purpose: str = "chat completion",
    **kwargs: Any,
) -> str:
    """Async chat completion returning text; provider failures become LLMError/RateLimitedError."""
    llm = model or get_chat_model()
    started = time.perf_counter()
    logger.info("Calling LLM for %s (model=%s)", purpose, model_name(llm))
    try:
        response = await llm.ainvoke(messages, config={"callbacks": callbacks or []}, **kwargs)
    except (ConfigError, LLMError):
        raise
    except Exception as exc:
        logger.warning("Async LLM call failed: %s", type(exc).__name__)
        raise map_llm_exception(exc) from exc
    content = response.content
    text = content if isinstance(content, str) else "".join(str(part) for part in content)
    if not text.strip():
        raise LLMError("The language model returned an empty answer.")
    logger.info("LLM call for %s succeeded in %.0f ms", purpose, (time.perf_counter() - started) * 1000)
    return text.strip()
