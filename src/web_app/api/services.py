"""Shared dependencies for the REST routers, injected through FastAPI dependencies (so tests can swap in fakes)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


from src.core.config import get_config
from src.core.errors import GuardrailError
from src.core.models import UserProfile
from src.core.pii import redact_text
from src.data.db import Database
from src.data.repositories import UserRepository
from src.utils.validators import validate_user_input
from src.web_app.errors import ApiError
from src.web_app.ratelimit import SlidingWindowLimiter


def default_limiters() -> dict[str, SlidingWindowLimiter]:
    cfg = get_config().api
    return {"read": SlidingWindowLimiter(cfg.read_per_minute), "expensive": SlidingWindowLimiter(cfg.expensive_per_minute),
            "chat": SlidingWindowLimiter(cfg.chat_per_minute)}


@dataclass
class ApiServices:
    db: Database
    market: Any                      # MarketService
    news: Any                        # NewsService
    portfolio_agent: Any             # PortfolioAgent (analyze() is deterministic; arun() narrates)
    goal_agent: Any                  # GoalPlanningAgent
    news_agent: Any                  # NewsAgent
    memory: Any                      # SqliteConversationMemory
    audio: Any = None                # AudioService (Whisper / TTS)
    limiters: dict[str, SlidingWindowLimiter] = field(default_factory=default_limiters)

    def require_user(self, user_id: str) -> UserProfile:
        profile = UserRepository(self.db).get(user_id)
        if profile is None:
            raise ApiError(404, "Unknown user id", "user_not_found")
        return profile

    async def sanitize(self, text: str, *, max_chars: int = 300) -> str:
        """Validate + PII-redact free text before it can reach an LLM or an external search (fail closed)."""
        try:
            cleaned = validate_user_input(text, max_chars)
        except GuardrailError as exc:
            raise ApiError(422, str(exc), exc.code) from exc
        try:
            return (await asyncio.to_thread(redact_text, cleaned)).text
        except Exception as exc:
            raise ApiError(503, "The privacy filter is unavailable, so the request was not sent.", "privacy_unavailable") from exc
