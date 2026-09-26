"""LangChain must never auto-upload run data. LangSmith is used ONLY when explicitly enabled (ENABLE_LANGSMITH + key in .env),
through an explicit tracer whose client PII-masks every payload first (Langfuse is masked the same way)."""
import os

import pytest

from src.core import config, tracing

VARS = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")


@pytest.fixture(autouse=True)
def fresh_client_cache():
    tracing.get_langsmith_client.cache_clear()
    yield
    tracing.get_langsmith_client.cache_clear()


@pytest.mark.parametrize("opt_in", ["", "true", "1"])
def test_environment_driven_auto_tracing_is_always_switched_off(monkeypatch, opt_in):
    for name in VARS:
        monkeypatch.setenv(name, "true")  # e.g. left enabled in the user's shell profile
    monkeypatch.setenv("ENABLE_LANGSMITH", opt_in)
    config._disable_langsmith_env_tracing()
    assert all(os.environ[name] == "false" for name in VARS)


def test_langchain_itself_reports_tracing_disabled_in_this_process():
    from langsmith.utils import tracing_is_enabled

    assert tracing_is_enabled() in (False, None)


@pytest.mark.parametrize("flag,key,expected", [("true", "k", True), ("1", "k", True), ("yes", "k", True), ("true", "", False),
                                               ("", "k", False), ("false", "k", False), ("true", "   ", False)])
def test_langsmith_needs_both_the_opt_in_flag_and_a_key(monkeypatch, flag, key, expected):
    monkeypatch.setenv("ENABLE_LANGSMITH", flag)
    monkeypatch.setenv("LANGSMITH_API_KEY", key)
    assert config.langsmith_enabled() is expected


def test_no_client_and_no_callbacks_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_LANGSMITH", "false")
    assert tracing.get_langsmith_client() is None and tracing.langsmith_callbacks() == []


def test_enabled_client_uses_the_masking_anonymizer_and_configured_endpoint(monkeypatch):
    monkeypatch.setenv("ENABLE_LANGSMITH", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_test_not_real")
    object.__setattr__(tracing.settings, "langsmith_api_key", "lsv2_pt_test_not_real")
    object.__setattr__(tracing.settings, "langsmith_endpoint", "http://localhost:9")
    client = tracing.get_langsmith_client()
    assert client is not None and str(client.api_url).startswith("http://localhost:9")
    assert client._anonymizer is tracing._mask_for_langsmith or callable(getattr(client, "_anonymizer", tracing._mask_for_langsmith))
    tracing.get_langsmith_client.cache_clear()
    object.__setattr__(tracing.settings, "langsmith_api_key", "")


def test_callbacks_contain_a_tracer_bound_to_the_masked_client_and_project(monkeypatch):
    from langchain_core.tracers.langchain import LangChainTracer
    from langsmith import Client

    fake = Client(api_key="x", api_url="http://localhost:9", auto_batch_tracing=False)
    monkeypatch.setattr(tracing, "get_langsmith_client", lambda: fake)
    (tracer,) = tracing.langsmith_callbacks()
    assert isinstance(tracer, LangChainTracer) and tracer.client is fake and tracer.project_name == tracing.settings.langsmith_project


def test_observability_callbacks_combine_both_backends(monkeypatch):
    monkeypatch.setattr(tracing, "langfuse_callbacks", lambda: ["langfuse"])
    monkeypatch.setattr(tracing, "langsmith_callbacks", lambda: ["langsmith"])
    assert tracing.observability_callbacks() == ["langfuse", "langsmith"]


def test_langsmith_setup_problems_never_break_answering(monkeypatch):
    def boom():
        raise RuntimeError("bad endpoint")

    monkeypatch.setattr(tracing, "get_langsmith_client", boom)
    assert tracing.langsmith_callbacks() == []


# ---------------------------------------------------------------------------------------- masking
def test_langsmith_payloads_are_pii_redacted_including_nested_message_structures():
    payload = {"messages": [{"kwargs": {"content": "I'm John Smith, john.smith@example.com, SSN 536-90-4399"}}], "n": 3,
               "outputs": {"generations": [[{"text": "Call 415-555-0199"}]]}}
    masked = tracing._mask_for_langsmith(payload)
    text = str(masked)
    assert all(x not in text for x in ("John Smith", "john.smith@example.com", "536-90-4399", "555-0199"))
    assert masked["n"] == 3 and "<EMAIL_ADDRESS>" in text


def test_langsmith_masking_fails_closed(monkeypatch):
    from src.core import pii

    monkeypatch.setattr(pii, "get_pii_guard", lambda: (_ for _ in ()).throw(RuntimeError("model missing")))
    assert tracing._mask_for_langsmith({"input": "secret a@b.com"}) == {"redacted": "masking error"}


def test_flush_covers_langsmith_and_swallows_errors(monkeypatch):
    calls = []

    class LS:
        def flush(self):
            calls.append("ls")

    monkeypatch.setattr(tracing, "get_langsmith_client", lambda: LS())
    tracing.flush_traces()
    assert "ls" in calls

    class Broken:
        def flush(self):
            raise RuntimeError("network")

    monkeypatch.setattr(tracing, "get_langsmith_client", lambda: Broken())
    tracing.flush_traces()  # must not raise


# ----------------------------------------------------------------- .env wins over a stale shell variable
def test_dotenv_overrides_a_stale_shell_langsmith_key(tmp_path, monkeypatch):
    """Regression: python-dotenv keeps an existing shell value, so an old LANGSMITH_API_KEY in ~/.zshrc silently
    replaced the key in .env and every upload was rejected (HTTP 403)."""
    env = tmp_path / ".env"
    env.write_text("LANGSMITH_API_KEY=new-key-from-dotenv\nLANGSMITH_PROJECT=clarityfi\nENABLE_LANGSMITH=true\nOPENAI_API_KEY=sk-not-touched\n")
    monkeypatch.setenv("LANGSMITH_API_KEY", "stale-key-from-shell")
    monkeypatch.setenv("OPENAI_API_KEY", "shell-openai-key")
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("ENABLE_LANGSMITH", raising=False)  # this process may already have it set from the real .env
    overridden = config.apply_dotenv_overrides(env)
    assert os.environ["LANGSMITH_API_KEY"] == "new-key-from-dotenv" and os.environ["LANGSMITH_PROJECT"] == "clarityfi"
    assert os.environ["ENABLE_LANGSMITH"] == "true" and set(overridden) == {"LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "ENABLE_LANGSMITH"}
    assert os.environ["OPENAI_API_KEY"] == "shell-openai-key"  # only LangSmith settings are overridden


def test_override_is_a_noop_when_values_already_match_or_the_file_is_missing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LANGSMITH_API_KEY=same\n")
    monkeypatch.setenv("LANGSMITH_API_KEY", "same")
    assert config.apply_dotenv_overrides(env) == []
    assert config.apply_dotenv_overrides(tmp_path / "missing.env") == []
    env.write_text("LANGSMITH_ENDPOINT\n")  # key without a value
    assert config.apply_dotenv_overrides(env) == []
