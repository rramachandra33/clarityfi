from datetime import date

import pytest
from pydantic import ValidationError

from src.core import llm
from src.core.disclaimers import ADVICE_REDIRECT, EDUCATIONAL_DISCLAIMER, with_disclaimer
from src.core.errors import ClarityError, ConfigError, LLMError, RateLimitedError
from src.core.models import AgentName, AgentResult, Goal, Holding, UserProfile


def test_holding_normalizes_ticker_and_rejects_negative_shares():
    assert Holding(ticker=" vti ", shares=2).ticker == "VTI"
    with pytest.raises(ValidationError):
        Holding(ticker="VTI", shares=-1)


def test_user_profile_validation():
    assert UserProfile(user_id="learner-204").risk_tolerance.value == "moderate"
    with pytest.raises(ValidationError):
        UserProfile(user_id="ab")
    with pytest.raises(ValidationError):
        UserProfile(user_id="bad;id<script>")


def test_goal_requires_positive_target():
    with pytest.raises(ValidationError):
        Goal(target_amount=0, target_date=date(2040, 1, 1))


def test_agent_result_ok_flag():
    assert AgentResult(agent=AgentName.tax, content="x").ok
    assert not AgentResult(agent=AgentName.tax, error="down").ok


def test_disclaimer_is_appended_once():
    once = with_disclaimer("Answer")
    assert EDUCATIONAL_DISCLAIMER in once
    assert with_disclaimer(once) == once
    assert "delayed" in with_disclaimer("Price", market_data=True)
    assert "can't tell you what to buy" in ADVICE_REDIRECT


def test_error_hierarchy_and_default_messages():
    assert issubclass(RateLimitedError, LLMError) and issubclass(LLMError, ClarityError)
    assert "busy" in str(RateLimitedError())


@pytest.fixture
def api_key():
    """Set settings.openai_api_key for a test (settings is a frozen dataclass) and restore it."""
    original = llm.settings.openai_api_key

    def _set(value):
        object.__setattr__(llm.settings, "openai_api_key", value)

    yield _set
    _set(original)


def test_get_chat_model_requires_key(api_key):
    api_key("")
    with pytest.raises(ConfigError):
        llm.get_chat_model()


def test_get_chat_model_uses_configured_params(api_key):
    api_key("test-key-not-real")
    model = llm.get_chat_model()
    assert model.model_name == llm.settings.openai_chat_model
    assert model.temperature == llm.get_config().llm.temperature


class _Err(Exception):
    def __init__(self, name, status=None):
        super().__init__(name)
        self.status_code = status


def _named(name, status=None):
    return type(name, (_Err,), {})(name, status)


@pytest.mark.parametrize(
    "exc,expected",
    [
        (_named("RateLimitError", 429), RateLimitedError),
        (_named("AuthenticationError", 401), LLMError),
        (_named("APITimeoutError"), LLMError),
        (_named("APIConnectionError"), LLMError),
        (RuntimeError("weird"), LLMError),
    ],
)
def test_map_llm_exception(exc, expected):
    assert isinstance(llm.map_llm_exception(exc), expected)


def test_invoke_chat_maps_provider_errors():
    class Boom:
        def invoke(self, *a, **k):
            raise _named("RateLimitError", 429)

    with pytest.raises(RateLimitedError):
        llm.invoke_chat("hi", model=Boom())


def test_invoke_chat_returns_text():
    class Msg:
        content = "hello"

    class Fake:
        def invoke(self, *a, **k):
            return Msg()

    assert llm.invoke_chat("hi", model=Fake()) == "hello"


# ---------------------------------------------------------------- Step 9 additions: async LLM, codes, disclaimer
import asyncio
from types import SimpleNamespace

from src.core.disclaimers import PLAIN_DISCLAIMER, ensure_educational_disclaimer
from src.core.errors import (GuardrailError as _G, KnowledgeBaseEmptyError, ProviderError, ProviderUnavailableError,
                             RetrievalError, SymbolNotFoundError)


def test_ainvoke_chat_returns_stripped_text_and_passes_callbacks():
    seen = {}

    class Fake:
        async def ainvoke(self, messages, config=None, **kw):
            seen["config"] = config
            return SimpleNamespace(content="  hello  ")

    assert asyncio.run(llm.ainvoke_chat("hi", model=Fake(), callbacks=["cb"])) == "hello" and seen["config"] == {"callbacks": ["cb"]}
    assert asyncio.run(llm.ainvoke_chat("hi", model=Fake())) == "hello"


def test_ainvoke_chat_joins_multipart_content_and_rejects_empty():
    class Parts:
        async def ainvoke(self, *a, **k):
            return SimpleNamespace(content=["a", "b"])

    class Blank:
        async def ainvoke(self, *a, **k):
            return SimpleNamespace(content="  ")

    assert asyncio.run(llm.ainvoke_chat("hi", model=Parts())) == "ab"
    with pytest.raises(LLMError, match="empty answer"):
        asyncio.run(llm.ainvoke_chat("hi", model=Blank()))


def test_ainvoke_chat_maps_provider_errors_and_passes_clarity_errors_through():
    class Rate:
        async def ainvoke(self, *a, **k):
            raise _named("RateLimitError", 429)

    class Own:
        async def ainvoke(self, *a, **k):
            raise ConfigError("no key")

    with pytest.raises(RateLimitedError):
        asyncio.run(llm.ainvoke_chat("hi", model=Rate()))
    with pytest.raises(ConfigError):
        asyncio.run(llm.ainvoke_chat("hi", model=Own()))


def test_every_error_class_has_a_distinct_stable_code():
    classes = [ClarityError, ConfigError, LLMError, RateLimitedError, RetrievalError, ProviderError, _G, SymbolNotFoundError,
               ProviderUnavailableError, KnowledgeBaseEmptyError]
    codes = [c.code for c in classes]
    assert len(set(codes)) == len(codes)
    assert KnowledgeBaseEmptyError.code == "knowledge_base_empty" and "empty" in str(KnowledgeBaseEmptyError())
    assert issubclass(KnowledgeBaseEmptyError, RetrievalError)


@pytest.mark.parametrize("text,changed", [
    ("An ETF is a basket.", True), ("An ETF is a basket. Educational only.", False),
    ("Note: educational information only.", False), ("This is not personalized financial advice.", False),
])
def test_ensure_educational_disclaimer(text, changed):
    out = ensure_educational_disclaimer(text)
    assert (out != text) is changed and (out.endswith(PLAIN_DISCLAIMER) if changed else out == text)
    assert ensure_educational_disclaimer(out) == out  # idempotent
    assert "_" not in PLAIN_DISCLAIMER and "*" not in PLAIN_DISCLAIMER  # plain text: safe for UI and voice


def test_model_name_falls_back_across_shapes():
    assert llm.model_name(SimpleNamespace(model_name="gpt-4o-mini")) == "gpt-4o-mini"
    assert llm.model_name(SimpleNamespace(model="gpt-4o")) == "gpt-4o"

    class NoName:
        pass

    assert llm.model_name(NoName()) == "NoName"


def test_invoke_chat_logs_the_model_and_call_purpose_at_info(caplog):
    class Fake:
        model_name = "gpt-4o-mini"

        def invoke(self, *a, **k):
            return SimpleNamespace(content="hello")

    with caplog.at_level("INFO"):
        llm.invoke_chat("hi", model=Fake(), purpose="unit test call")
    assert any("Calling LLM for unit test call (model=gpt-4o-mini)" in m for m in caplog.messages)
    assert any(m.startswith("LLM call for unit test call succeeded in") for m in caplog.messages)


def test_ainvoke_chat_logs_the_model_and_call_purpose_at_info(caplog):
    class Fake:
        model_name = "gpt-4o-mini"

        async def ainvoke(self, *a, **k):
            return SimpleNamespace(content="hello")

    with caplog.at_level("INFO"):
        asyncio.run(llm.ainvoke_chat("hi", model=Fake(), purpose="unit test call"))
    assert any("Calling LLM for unit test call (model=gpt-4o-mini)" in m for m in caplog.messages)
    assert any(m.startswith("LLM call for unit test call succeeded in") for m in caplog.messages)


def test_ainvoke_chat_does_not_log_success_when_the_answer_is_empty(caplog):
    class Blank:
        async def ainvoke(self, *a, **k):
            return SimpleNamespace(content="  ")

    with caplog.at_level("INFO"):
        with pytest.raises(LLMError):
            asyncio.run(llm.ainvoke_chat("hi", model=Blank(), purpose="unit test call"))
    assert not any("succeeded" in m for m in caplog.messages)
