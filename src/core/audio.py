"""Speech services: OpenAI Whisper (speech-to-text) and text-to-speech.

Privacy: audio is processed in memory and never written to disk, logged, or traced; only sizes are logged.
The browser's built-in voice remains the free fallback, so every failure here is a plain `AudioError`.
"""
from __future__ import annotations

import logging
from typing import Any

from src.core.config import get_config, settings
from src.core.errors import AudioError, ConfigError

logger = logging.getLogger(__name__)

EXTENSIONS = {"webm": "webm", "ogg": "ogg", "mp4": "mp4", "m4a": "m4a", "x-m4a": "m4a", "mpeg": "mp3", "mp3": "mp3",
              "wav": "wav", "x-wav": "wav", "wave": "wav"}


def base_type(content_type: str | None) -> str:
    """'audio/webm;codecs=opus' -> 'audio/webm'."""
    return (content_type or "").split(";")[0].strip().lower()


def filename_for(content_type: str) -> str:
    return f"speech.{EXTENSIONS.get(base_type(content_type).split('/')[-1], 'webm')}"


class AudioService:
    def __init__(self, client: Any = None) -> None:
        self._client = client

    @property
    def available(self) -> bool:
        return self._client is not None or bool(settings.openai_api_key)

    @property
    def client(self):
        if self._client is None:
            if not settings.openai_api_key:
                raise ConfigError("OPENAI_API_KEY is not set. Add it to .env.")
            from openai import AsyncOpenAI

            cfg = get_config().audio
            self._client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url,
                                       timeout=cfg.timeout_seconds, max_retries=1)
        return self._client

    async def transcribe(self, data: bytes, content_type: str) -> str:
        cfg = get_config().audio
        try:
            result = await self.client.audio.transcriptions.create(
                model=cfg.stt_model, file=(filename_for(content_type), data, base_type(content_type)), language="en")
        except ConfigError as exc:
            raise AudioError(str(exc)) from exc
        except Exception as exc:  # provider/network errors: never include details that could echo content
            logger.warning("transcription failed (%s), %d bytes", type(exc).__name__, len(data))
            raise AudioError() from exc
        text = result if isinstance(result, str) else getattr(result, "text", "")
        return (text or "").strip()

    async def speak(self, text: str) -> bytes:
        cfg = get_config().audio
        try:
            response = await self.client.audio.speech.create(model=cfg.tts_model, voice=cfg.voice, input=text, response_format="mp3")
            audio = response.content if hasattr(response, "content") else await response.aread()
        except ConfigError as exc:
            raise AudioError(str(exc)) from exc
        except Exception as exc:
            logger.warning("speech synthesis failed (%s), %d chars", type(exc).__name__, len(text))
            raise AudioError() from exc
        if not audio:
            raise AudioError()
        return audio
