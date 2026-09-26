"""Add the ClarityFI MCP server to Claude Desktop's configuration (you run this; nothing else edits that file).

    python3 scripts/install_claude_desktop.py --dry-run     # show what would change
    python3 scripts/install_claude_desktop.py               # back up the config, then add/update the "clarityfi" entry

It keeps every other setting and every other MCP server exactly as they are, writes a timestamped backup next to the
file first, and stores no secrets (the server reads keys from the project's .env). Restart Claude Desktop afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

DEFAULT_CONFIG = Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def server_entry(root: Path = PROJECT_ROOT) -> dict:
    return {"command": str(root / ".venv/bin/python"), "args": ["-m", "src.mcp_server.server"], "cwd": str(root)}


def merged(config: dict, root: Path = PROJECT_ROOT) -> dict:
    """The config with the clarityfi entry added or updated; everything else untouched."""
    out = dict(config)
    servers = dict(out.get("mcpServers") or {})
    servers["clarityfi"] = server_entry(root)
    out["mcpServers"] = servers
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Claude Desktop config file (default: the standard macOS location)")
    parser.add_argument("--dry-run", action="store_true", help="print the result without writing anything")
    args = parser.parse_args(argv)

    if not (PROJECT_ROOT / ".venv/bin/python").exists():
        print("The project's virtual environment was not found. Run setup_mac.command first.", file=sys.stderr)
        return 1
    config: dict = {}
    if args.config.exists():
        try:
            config = json.loads(args.config.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as exc:
            print(f"{args.config} is not valid JSON ({exc}). Nothing was changed.", file=sys.stderr)
            return 1
    result = merged(config)
    existing = sorted((config.get("mcpServers") or {}).keys())
    print(f"Existing MCP servers kept: {existing or 'none'}")
    print(json.dumps({"mcpServers": {"clarityfi": result["mcpServers"]["clarityfi"]}}, indent=2))
    if args.dry_run:
        print("(dry run: nothing written)")
        return 0
    args.config.parent.mkdir(parents=True, exist_ok=True)
    if args.config.exists():
        backup = args.config.with_name(f"{args.config.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(args.config, backup)
        os.chmod(backup, 0o600)
        print(f"Backup written: {backup}")
    tmp = args.config.with_name(args.config.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, args.config)
    print("Done. Quit and reopen Claude Desktop, then look for the 'clarityfi' tools.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
