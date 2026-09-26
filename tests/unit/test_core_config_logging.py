import io
import json
import logging

import pytest

from src.core.config import AppConfig, load_app_config
from src.utils.logging import TIMESTAMP_FORMAT, JsonFormatter, ContextFilter, log_context, mask_secrets, setup_logging


def test_yaml_config_loads_repo_defaults():
    cfg = load_app_config()
    assert cfg.cache.market_ttl_seconds == 1800
    assert cfg.market.providers == ["alphavantage", "yfinance", "mock"]
    assert cfg.logging.json_output is True
    assert 0 < cfg.cache.qa_semantic_threshold <= 1 and cfg.cache.qa_semantic_threshold == 0.90 and cfg.cache.qa_enabled


def test_missing_yaml_falls_back_to_defaults(tmp_path):
    cfg = load_app_config(tmp_path / "nope.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.rag.top_k == 3


def test_partial_yaml_merges_with_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("cache:\n  market_ttl_seconds: 60\n")
    cfg = load_app_config(path)
    assert cfg.cache.market_ttl_seconds == 60
    assert cfg.cache.news_ttl_seconds == 1800


def test_rag_top_k_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "7")
    assert load_app_config(tmp_path / "none.yaml").rag.top_k == 7


def test_invalid_yaml_value_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("llm:\n  max_retries: not-a-number\n")
    with pytest.raises(Exception):
        load_app_config(path)


@pytest.mark.parametrize(
    "secret",
    [
        "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
        "sk-lf-00000000-1111-2222-3333-444444444444",
        "pk-lf-55555555-6666-7777-8888-999999999999",
    ],
)
def test_mask_secrets_hides_key_shapes(secret):
    assert secret not in mask_secrets(f"failed with key {secret} today")


def test_mask_secrets_hides_assigned_values():
    masked = mask_secrets("api_key=abcdef123456789 and SERPAPI token: zzzzzzzzzz1234")
    assert "abcdef123456789" not in masked
    assert "zzzzzzzzzz1234" not in masked


def _logger_with_stream():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(ContextFilter())
    handler.setFormatter(JsonFormatter())
    log = logging.getLogger("test.clarity.json")
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.INFO)
    return log, stream


def _split_line(line: str) -> tuple[str, dict]:
    """Every log line is `<19-char timestamp> <json object>`; split and parse both parts."""
    timestamp, payload = line[:19], line[20:]
    return timestamp, json.loads(payload)


def test_json_log_has_context_and_masks_secrets():
    log, stream = _logger_with_stream()
    with log_context(request_id="req-1", session_id="s-9"):
        log.info("calling with %s", "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789")
    _ts, record = _split_line(stream.getvalue())
    assert record["request_id"] == "req-1" and record["session_id"] == "s-9"
    assert record["level"] == "INFO"
    assert "abcdefghijklmnopqrstuvwxyz" not in record["message"]
    assert "ts" not in record  # the timestamp is the line's leading prefix, not duplicated inside the JSON


def test_password_reset_email_logger_is_exempt_from_secret_masking():
    """The console-mode dev email sender (src.core.email) deliberately logs a reset link containing
    "reset_token=...". The generic api_key/secret/token masking pattern would otherwise eat that token,
    breaking the whole point of console mode: letting a developer read the reset link without real SES."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(ContextFilter())
    handler.setFormatter(JsonFormatter())
    log = logging.getLogger("src.core.email")
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.INFO)
    log.info("reset link: http://127.0.0.1:5173/?reset_token=abcdef0123456789")
    _ts, record = _split_line(stream.getvalue())
    assert "reset_token=abcdef0123456789" in record["message"]


def test_log_context_is_scoped():
    log, stream = _logger_with_stream()
    with log_context(request_id="inner"):
        pass
    log.info("after")
    _ts, record = _split_line(stream.getvalue())
    assert "request_id" not in record


def test_exception_is_logged_as_json():
    log, stream = _logger_with_stream()
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("failed")
    _ts, record = _split_line(stream.getvalue())
    assert "ValueError" in record["exception"]


def test_setup_logging_is_idempotent():
    setup_logging("INFO", True)
    setup_logging("DEBUG", False)
    root = logging.getLogger()
    assert sum(1 for h in root.handlers if getattr(h, "_clarity", False)) == 1
    setup_logging("INFO", True)


def test_settings_repr_never_shows_secrets():
    from src.core.config import Settings

    text = repr(Settings(openai_api_key="sk-test-secret", serpapi_api_key="serp-secret", alphavantage_api_key="av-secret",
                         langfuse_secret_key="lf-secret", langfuse_public_key="lf-public", langsmith_api_key="ls-secret"))
    for secret in ("sk-test-secret", "serp-secret", "av-secret", "lf-secret", "lf-public", "ls-secret"):
        assert secret not in text


def test_timestamp_format_is_yyyy_mm_dd_hh24_mi_ss():
    import re

    log, stream = _logger_with_stream()
    log.info("hello")
    line = stream.getvalue()
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \{", line)  # the timestamp leads every line, JSON or not
    ts, record = _split_line(line)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", ts) and record["message"] == "hello"


def test_json_formatter_uses_local_time_like_uvicorn_and_the_text_formatter():
    """Regression test: the JSON formatter once stamped lines in UTC while uvicorn's own lines (and the
    plain-text formatter) use the system's local time zone, so the same request showed two different
    clock times across clarity-api.log. Both must agree."""
    import time

    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None)
    record.created = 1_700_000_000.0  # an arbitrary fixed instant
    stamped = JsonFormatter().format(record)
    expected = time.strftime(TIMESTAMP_FORMAT, time.localtime(record.created))
    assert stamped.startswith(expected)


def test_text_format_uses_the_same_timestamp_format():
    import re

    from src.utils.logging import TIMESTAMP_FORMAT, _TEXT_FORMAT

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(ContextFilter())
    handler.setFormatter(logging.Formatter(_TEXT_FORMAT, datefmt=TIMESTAMP_FORMAT))
    log = logging.getLogger("test.clarity.text")
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.INFO)
    log.info("hello")
    prefix = stream.getvalue().split(" ", 2)
    stamp = f"{prefix[0]} {prefix[1]}"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", stamp)


def test_uvicorn_log_config_uses_the_same_timestamp_format():
    """uvicorn's own startup/access lines (its own loggers, never our root logger) must carry the same
    YYYY-MM-DD HH:MM:SS stamp as the app's own logs, so nothing in clarity-api.log is left unstamped."""
    import json

    from src.core.config import PROJECT_ROOT
    from src.utils.logging import TIMESTAMP_FORMAT

    cfg = json.loads((PROJECT_ROOT / "uvicorn_log_config.json").read_text())
    for name in ("default", "access"):
        formatter = cfg["formatters"][name]
        assert formatter["datefmt"] == TIMESTAMP_FORMAT
        assert "%(asctime)s" in formatter["fmt"]
    for logger_name in ("uvicorn", "uvicorn.access"):
        assert cfg["loggers"][logger_name]["propagate"] is False  # must not double-log through our root handler
