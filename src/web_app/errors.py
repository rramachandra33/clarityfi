"""One place that maps application errors to HTTP responses (used by the chat endpoint and every router)."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.errors import ClarityError

STATUS_BY_CODE: dict[str, int] = {
    "knowledge_base_empty": 409,
    "symbol_not_found": 404,
    "guardrail_rejected": 422,
    "rate_limited": 503,           # the UPSTREAM provider is rate limiting us (our own limiter answers 429)
    "llm_unavailable": 503,
    "audio_unavailable": 503,
    "config_error": 503,
    "retrieval_error": 503,
    "provider_unavailable": 503,
    "privacy_unavailable": 503,
    "agent_timeout": 504,
}


def status_for(code: str | None) -> int:
    return STATUS_BY_CODE.get(code or "", 500)


def error_body(message: str, code: str | None) -> dict[str, str | None]:
    return {"detail": message, "code": code}


class ApiError(Exception):
    """An HTTP error with a flat body {"detail": message, "code": code} (optionally with headers, e.g. Retry-After)."""

    def __init__(self, status_code: int, message: str, code: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status_code, self.message, self.code, self.headers = status_code, message, code, headers


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=error_body(exc.message, exc.code), headers=exc.headers)

    @app.exception_handler(ClarityError)
    async def clarity_error_handler(_request: Request, exc: ClarityError) -> JSONResponse:
        """Expected application errors -> a readable message and a stable machine-readable code."""
        return JSONResponse(status_code=status_for(exc.code), content=error_body(str(exc), exc.code))
