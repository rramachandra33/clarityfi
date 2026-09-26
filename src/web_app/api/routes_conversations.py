"""Conversation history endpoints, so the UI can restore a chat after a reload."""
from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, Path, Query, Request

from src.data.repositories import MessageRepository, SessionRepository
from src.web_app.api.models import NOT_FOUND, ConversationOut, MessageOut, UserIdPath
from src.web_app.api.services import ApiServices
from src.web_app.ratelimit import client_key, enforce

ConversationPath = Path(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._:\-]+$", description="The client's conversation id")


def build_conversations_router(get_services: Callable[[], ApiServices]) -> APIRouter:
    router = APIRouter(prefix="/api/conversations", tags=["conversations"])

    @router.get("/{user_id}/{conversation_id}", response_model=ConversationOut, responses=NOT_FOUND,
                summary="Saved messages of one conversation (redacted text, oldest first)")
    def get_conversation(user_id: UserIdPath, request: Request, conversation_id: str = ConversationPath,
                         limit: int = Query(default=50, ge=1, le=200), svc: ApiServices = Depends(get_services)) -> ConversationOut:
        enforce(svc.limiters["read"], client_key(request))
        profile = svc.require_user(user_id)
        session_id = SessionRepository(svc.db).find(profile.user_id, conversation_id)
        rows = MessageRepository(svc.db).history(session_id, limit) if session_id else []
        return ConversationOut(user_id=profile.user_id, conversation_id=conversation_id, messages=[MessageOut(**r) for r in rows])

    @router.delete("/{user_id}/{conversation_id}", status_code=204, responses=NOT_FOUND, summary="Delete a conversation's saved messages")
    def clear_conversation(user_id: UserIdPath, request: Request, conversation_id: str = ConversationPath,
                           svc: ApiServices = Depends(get_services)) -> None:
        enforce(svc.limiters["read"], client_key(request))
        profile = svc.require_user(user_id)
        svc.memory.clear(svc.memory.key(profile.user_id, conversation_id))

    return router
