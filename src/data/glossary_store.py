from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GlossaryEntry:
    term: str
    definition: str


def _extract_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("term"), str) and isinstance(data.get("definition"), str):
            return [data]
        for key in ("glossary", "items", "terms", "data", "records"):
            if isinstance(data.get(key), list):
                return [item for item in data[key] if isinstance(item, dict)]
    raise ValueError("Expected a glossary array or an object containing a glossary/items/terms/data array")


def load_glossary_file(file_path: str | Path) -> list[GlossaryEntry]:
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Glossary file not found: {path}")
    if path.suffix.lower() not in {".json", ".jsonl", ".ndjson"}:
        raise ValueError("Expected a .json, .jsonl, or .ndjson glossary file")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("The glossary file is empty")

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        raw_records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            if isinstance(value, dict):
                raw_records.append(value)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Also accept a comma-separated sequence of objects like the supplied example.
            try:
                data = json.loads(f"[{text.rstrip(',')}]")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid glossary JSON: {exc.msg} at line {exc.lineno}") from exc
        raw_records = _extract_records(data)

    entries: list[GlossaryEntry] = []
    seen: set[str] = set()
    for index, record in enumerate(raw_records, start=1):
        term = record.get("term")
        definition = record.get("definition")
        if not isinstance(term, str) or not term.strip():
            raise ValueError(f"Glossary item {index} is missing a non-empty 'term'")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(f"Glossary item {index} is missing a non-empty 'definition'")
        normalized = " ".join(term.split()).casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        entries.append(
            GlossaryEntry(
                term=" ".join(term.split()),
                definition=" ".join(definition.split()),
            )
        )
    if not entries:
        raise ValueError("No valid glossary items were found")
    return entries


class GlossaryAlreadyInitializedError(RuntimeError):
    pass


class GlossaryStore:
    """Immutable-after-initialization glossary backed by local SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS glossary (
                    term TEXT PRIMARY KEY COLLATE NOCASE,
                    definition TEXT NOT NULL
                )
                """
            )

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM glossary").fetchone()
            return int(row["count"])

    def initialize(self, entries: list[GlossaryEntry]) -> int:
        if not entries:
            raise ValueError("No glossary items were supplied")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = int(connection.execute("SELECT COUNT(*) FROM glossary").fetchone()[0])
            if existing:
                raise GlossaryAlreadyInitializedError(
                    f"Glossary is already initialized with {existing} terms and is read-only"
                )
            connection.executemany(
                "INSERT INTO glossary (term, definition) VALUES (?, ?)",
                [(entry.term, entry.definition) for entry in entries],
            )
            return len(entries)

    def initialize_from_file(self, file_path: str | Path) -> int:
        return self.initialize(load_glossary_file(file_path))

    def list(self, query: str = "", limit: int = 1000) -> list[GlossaryEntry]:
        safe_limit = max(1, min(limit, 5000))
        with self._connect() as connection:
            if query.strip():
                pattern = f"%{query.strip()}%"
                rows = connection.execute(
                    """
                    SELECT term, definition FROM glossary
                    WHERE term LIKE ? COLLATE NOCASE OR definition LIKE ? COLLATE NOCASE
                    ORDER BY term COLLATE NOCASE
                    LIMIT ?
                    """,
                    (pattern, pattern, safe_limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT term, definition FROM glossary ORDER BY term COLLATE NOCASE LIMIT ?",
                    (safe_limit,),
                ).fetchall()
        return [GlossaryEntry(term=row["term"], definition=row["definition"]) for row in rows]
