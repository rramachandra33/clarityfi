"""Audio endpoints and service: validation, limits, privacy, and graceful failure (no real OpenAI calls)."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.core import audio as audio_mod
from src.core.audio import AudioService, base_type, filename_for
from src.core.errors import AudioError
from src.data.db import Database
from src.data.seed import seed
from src.web_app.api import main
from src.web_app.api.services import ApiServices, default_limiters
from tests.unit._agent_fakes import FakeMarket


class FakeOpenAI:
    def __init__(self, transcript="what is an etf", audio=b"ID3fakemp3", error=None):
        self.error, self.transcript, self.audio = error, transcript, audio
        self.calls = []
        self.audio_ns = SimpleNamespace(transcriptions=SimpleNamespace(create=self._stt), speech=SimpleNamespace(create=self._tts))

    @property
    def audio(self):
        return self.audio_ns

    @audio.setter
    def audio(self, value):
        self._audio_bytes = value

    async def _stt(self, **kw):
        self.calls.append(("stt", kw))
        if self.error:
            raise self.error
        return SimpleNamespace(text=f"  {self.transcript} ")

    async def _tts(self, **kw):
        self.calls.append(("tts", kw))
        if self.error:
            raise self.error
        return SimpleNamespace(content=self._audio_bytes)


@pytest.fixture
def client(tmp_path):
    db = Database(tmp_path / "a.db")
    db.init_schema()
    seed(db)
    fake = FakeOpenAI()
    svc = ApiServices(db=db, market=FakeMarket({}), news=None, portfolio_agent=None, goal_agent=None, news_agent=None, memory=None,
                      audio=AudioService(fake), limiters=default_limiters())
    main.app.dependency_overrides[main.get_services] = lambda: svc
    with TestClient(main.app) as c:
        c.svc, c.fake = svc, fake
        yield c
    main.app.dependency_overrides.clear()


def upload(client, data=b"\x1aE\xdf\xa3audio", kind="audio/webm;codecs=opus"):
    return client.post("/api/audio/transcribe", files={"file": ("rec.webm", data, kind)})


def test_transcribe_returns_trimmed_text_and_sends_language(client):
    r = upload(client)
    assert r.status_code == 200 and r.json() == {"text": "what is an etf"}
    kw = client.fake.calls[0][1]
    assert kw["language"] == "en" and kw["file"][0] == "speech.webm" and kw["file"][2] == "audio/webm"


@pytest.mark.parametrize("kind,code,status", [("text/plain", "unsupported_audio_type", 415), ("application/pdf", "unsupported_audio_type", 415)])
def test_transcribe_rejects_wrong_types(client, kind, code, status):
    r = upload(client, kind=kind)
    assert r.status_code == status and r.json()["code"] == code and not client.fake.calls


def test_transcribe_rejects_empty_and_oversized(client, monkeypatch):
    assert upload(client, data=b"").json()["code"] == "audio_empty"
    monkeypatch.setattr(main.get_config().audio, "max_upload_bytes", 10)
    r = upload(client, data=b"x" * 11)
    assert r.status_code == 413 and r.json()["code"] == "audio_too_large" and not client.fake.calls


def test_provider_failure_is_a_clear_503_and_leaks_nothing(client):
    client.fake.error = RuntimeError("boom sk-secret-value")
    r = upload(client)
    assert r.status_code == 503 and r.json()["code"] == "audio_unavailable" and "sk-secret" not in r.text


def test_speak_returns_mp3_and_redacts_pii_before_sending(client):
    r = client.post("/api/audio/speak", json={"text": "Email me at jane.doe@example.com about ETFs."})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and r.content == b"ID3fakemp3"
    assert r.headers["cache-control"] == "no-store"
    sent = client.fake.calls[0][1]["input"]
    assert "jane.doe@example.com" not in sent and "ETFs" in sent


def test_speak_validates_length_and_blank(client):
    assert client.post("/api/audio/speak", json={"text": "a" * 1600}).status_code == 422
    assert client.post("/api/audio/speak", json={"text": "   "}).status_code == 422
    assert not client.fake.calls


def test_speak_fails_closed_when_privacy_filter_breaks(client, monkeypatch):
    def boom(_):
        raise RuntimeError("nlp down")
    monkeypatch.setattr("src.web_app.api.routes_audio.redact_output", boom)
    r = client.post("/api/audio/speak", json={"text": "hello world"})
    assert r.status_code == 503 and r.json()["code"] == "privacy_unavailable" and not client.fake.calls


def test_status_and_missing_service(client):
    assert client.get("/api/audio/status").json()["transcription"] is True
    client.svc.audio = None
    assert client.get("/api/audio/status").json()["speech"] is False
    assert upload(client).json()["code"] == "audio_unavailable"
    assert client.post("/api/audio/speak", json={"text": "hi there"}).status_code == 503


def test_rate_limited(client):
    from src.web_app.ratelimit import SlidingWindowLimiter
    client.svc.limiters["expensive"] = SlidingWindowLimiter(1)
    assert client.post("/api/audio/speak", json={"text": "hi there"}).status_code == 200
    assert client.post("/api/audio/speak", json={"text": "hi there"}).status_code == 429


def test_helpers_and_service_edges(monkeypatch):
    assert base_type("Audio/WebM; codecs=opus") == "audio/webm" and base_type(None) == ""
    assert filename_for("audio/mpeg") == "speech.mp3" and filename_for("audio/x-m4a") == "speech.m4a" and filename_for("audio/weird") == "speech.webm"
    monkeypatch.setattr(audio_mod, "settings", SimpleNamespace(openai_api_key=""))
    svc = AudioService()
    assert svc.available is False
    with pytest.raises(AudioError):
        asyncio.run(svc.transcribe(b"x", "audio/webm"))
    with pytest.raises(AudioError):
        asyncio.run(svc.speak("hi"))
    empty = AudioService(FakeOpenAI(audio=b""))
    with pytest.raises(AudioError):
        asyncio.run(empty.speak("hi"))
    str_result = FakeOpenAI()
    async def as_str(**kw):
        return "plain text"
    str_result.audio_ns.transcriptions.create = as_str
    assert asyncio.run(AudioService(str_result).transcribe(b"x", "audio/wav")) == "plain text"


def test_service_builds_a_real_client_lazily_and_llm_wrappers_need_a_key(monkeypatch):
    from src.core import llm as llm_mod
    from src.core.errors import ConfigError

    monkeypatch.setattr(audio_mod, "settings", SimpleNamespace(openai_api_key="test-key", openai_base_url="https://api.example.com/v1"))
    svc = AudioService()
    assert svc.available and svc.client is svc.client  # built once, cached
    monkeypatch.setattr(llm_mod, "settings", SimpleNamespace(openai_api_key="", openai_embed_model="e", openai_base_url="https://x", openai_chat_model="c"))
    with pytest.raises(ConfigError):
        llm_mod.get_embeddings()
    monkeypatch.setattr(llm_mod, "settings", SimpleNamespace(openai_api_key="test-key", openai_embed_model="text-embedding-3-small",
                                                             openai_base_url="https://api.example.com/v1", openai_chat_model="gpt-4o-mini"))
    assert llm_mod.get_embeddings(model="text-embedding-3-small") is not None
