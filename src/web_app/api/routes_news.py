"""News endpoint: recent headlines (deterministic search) with an optional narrated summary."""
from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, Query, Request

from src.agents.base import AgentRequest
from src.agents.news import news_query
from src.web_app.api.models import TOO_MANY, NewsOut
from src.web_app.api.services import ApiServices
from src.web_app.ratelimit import client_key, enforce


def build_news_router(get_services: Callable[[], ApiServices]) -> APIRouter:
    router = APIRouter(prefix="/api/news", tags=["news"])

    @router.get("", response_model=NewsOut, responses=TOO_MANY, summary="Recent headlines about a company, market, or topic",
                description="Providers: SerpAPI Google News, then yFinance news, then the last saved headlines, then clearly-labelled SAMPLE "
                            "headlines. The query is PII-redacted before it leaves the machine. Cached / last-known / SAMPLE results carry a "
                            "`notice`. `summarize=true` also returns a narrated summary (uses the LLM).")
    async def news(request: Request, q: str = Query(default="", max_length=200, description="Company, ticker or topic"),
                   limit: int = Query(default=6, ge=1, le=10), summarize: bool = Query(default=False),
                   svc: ApiServices = Depends(get_services)) -> NewsOut:
        enforce(svc.limiters["expensive"], client_key(request))
        query = news_query(await svc.sanitize(q, max_chars=200)) if q.strip() else news_query("")
        import asyncio

        items, meta = await asyncio.to_thread(svc.news.search, query, limit)
        summary = None
        if summarize and meta.get("data_status") != "sample" and items:
            result = await svc.news_agent.arun(AgentRequest(question=query, user_id="anonymous"))
            summary = result.content if result.ok else None
        return NewsOut(query=query, items=items, source=meta.get("source"), data_status=meta.get("data_status", "live"),
                       notice=meta.get("notice"), stale=bool(meta.get("stale")), cache_hit=bool(meta.get("cache_hit")), summary=summary)

    return router
