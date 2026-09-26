import json

import pytest

from scripts import install_claude_desktop as installer


def run(tmp_path, capsys, *args):
    cfg = tmp_path / "claude_desktop_config.json"
    return cfg, lambda: installer.main(["--config", str(cfg), *args])


def test_merge_keeps_everything_else_and_is_idempotent():
    original = {"mcpServers": {"other": {"command": "x"}}, "preferences": {"a": 1}}
    once = installer.merged(original)
    assert once["mcpServers"]["other"] == {"command": "x"} and once["preferences"] == {"a": 1}
    assert once["mcpServers"]["clarityfi"]["args"] == ["-m", "src.mcp_server.server"] and "env" not in once["mcpServers"]["clarityfi"]
    assert installer.merged(once) == once and "clarityfi" not in original["mcpServers"]  # input not mutated
    assert "clarityfi" in installer.merged({})["mcpServers"]


def test_dry_run_writes_nothing_and_real_run_backs_up_first(tmp_path, capsys):
    cfg, go = run(tmp_path, capsys, "--dry-run")
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    before = cfg.read_text()
    assert go() == 0 and cfg.read_text() == before and "dry run" in capsys.readouterr().out
    cfg2, go2 = run(tmp_path, capsys)
    assert go2() == 0
    data = json.loads(cfg.read_text())
    assert set(data["mcpServers"]) == {"other", "clarityfi"}
    backups = list(tmp_path.glob("claude_desktop_config.json.bak-*"))
    assert len(backups) == 1 and json.loads(backups[0].read_text()) == json.loads(before)
    assert oct(cfg.stat().st_mode)[-3:] == "600" and not (tmp_path / "claude_desktop_config.json.tmp").exists()


def test_creates_a_missing_config_and_refuses_invalid_json(tmp_path, capsys):
    cfg = tmp_path / "sub" / "claude_desktop_config.json"
    assert installer.main(["--config", str(cfg)]) == 0 and "clarityfi" in json.loads(cfg.read_text())["mcpServers"]
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    assert installer.main(["--config", str(bad)]) == 1 and bad.read_text() == "{oops"
    assert "not valid JSON" in capsys.readouterr().err


def test_requires_the_project_virtualenv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(installer, "PROJECT_ROOT", tmp_path)
    assert installer.main(["--config", str(tmp_path / "c.json")]) == 1 and "virtual environment" in capsys.readouterr().err
