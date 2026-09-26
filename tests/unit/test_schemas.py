import pytest
from pydantic import ValidationError

from src.web_app.schemas import ChatRequest


def test_chat_request_accepts_voice_mode():
    request = ChatRequest(
        user_id="learner-204",
        question="How do ETFs work?",
        input_mode="voice",
        response_mode="voice",
    )
    assert request.input_mode == "voice"
    assert request.response_mode == "voice"


def test_chat_request_rejects_unknown_voice_mode():
    with pytest.raises(ValidationError):
        ChatRequest(user_id="learner-204", question="What is a bond?", input_mode="audio")
