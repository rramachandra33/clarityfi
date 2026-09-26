"""Audio endpoints: Whisper speech-to-text and text-to-speech. The browser's own voice is the fallback (see the UI)."""
from __future__ import annotations

import asyncio
from typing import Callable

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from pydantic import BaseModel, Field

from src.core.config import get_config
from src.core.errors import AudioError, GuardrailError
from src.core.pii import redact_output
from src.utils.validators import validate_user_input
from src.web_app.api.models import TOO_MANY
from src.web_app.api.services import ApiServices
from src.web_app.errors import ApiError
from src.web_app.ratelimit import client_key, enforce

from src.core.audio import base_type


class AudioStatus(BaseModel):
    transcription: bool
    speech: bool
    max_upload_bytes: int
    max_speech_chars: int


class TranscriptOut(BaseModel):
    text: str = Field(description="What was heard; empty when nothing intelligible was said")


class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


def build_audio_router(get_services: Callable[[], ApiServices]) -> APIRouter:
    router = APIRouter(prefix="/api/audio", tags=["audio"])

    @router.get("/status", response_model=AudioStatus, summary="Whether higher-quality speech services are available")
    def status(request: Request, svc: ApiServices = Depends(get_services)) -> AudioStatus:
        enforce(svc.limiters["read"], client_key(request))
        cfg = get_config().audio
        ok = bool(svc.audio and svc.audio.available)
        return AudioStatus(transcription=ok, speech=ok, max_upload_bytes=cfg.max_upload_bytes, max_speech_chars=cfg.max_tts_chars)

    @router.post("/transcribe", response_model=TranscriptOut, responses={**TOO_MANY, 413: {"description": "Audio too large"},
                 415: {"description": "Unsupported audio type"}}, summary="Speech to text (Whisper)",
                 description="Send a short recording as multipart `file`. The transcript is returned to the caller only; send it to "
                             "/api/chat as a normal question so it is PII-redacted like typed text. Audio is never stored.")
    async def transcribe(request: Request, file: UploadFile = File(...), svc: ApiServices = Depends(get_services)) -> TranscriptOut:
        enforce(svc.limiters["expensive"], client_key(request))
        cfg = get_config().audio
        kind = base_type(file.content_type)
        if kind not in cfg.allowed_types:
            raise ApiError(415, "Unsupported audio type. Use webm, ogg, mp4/m4a, mp3 or wav.", "unsupported_audio_type")
        data = await file.read(cfg.max_upload_bytes + 1)
        if len(data) > cfg.max_upload_bytes:
            raise ApiError(413, f"Recording is too large (max {cfg.max_upload_bytes // 1_000_000} MB). Try a shorter question.", "audio_too_large")
        if not data:
            raise ApiError(422, "The recording was empty.", "audio_empty")
        if svc.audio is None:
            raise AudioError()
        return TranscriptOut(text=await svc.audio.transcribe(data, kind))

    @router.post("/speak", responses={**TOO_MANY, 200: {"content": {"audio/mpeg": {}}, "description": "MP3 audio"}},
                 response_class=Response, summary="Text to speech",
                 description="Returns MP3. Text is validated and PII-filtered (fail closed) before it leaves the machine.")
    async def speak(body: SpeakIn, request: Request, svc: ApiServices = Depends(get_services)) -> Response:
        enforce(svc.limiters["expensive"], client_key(request))
        cfg = get_config().audio
        try:
            cleaned = validate_user_input(body.text, cfg.max_tts_chars)
        except GuardrailError as exc:
            raise ApiError(422, str(exc), exc.code) from exc
        try:
            safe = (await asyncio.to_thread(redact_output, cleaned)).text
        except Exception as exc:
            raise ApiError(503, "The privacy filter is unavailable, so the text was not sent.", "privacy_unavailable") from exc
        if svc.audio is None:
            raise AudioError()
        audio = await svc.audio.speak(safe)
        return Response(content=audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})

    return router
