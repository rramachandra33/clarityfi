from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


TEXT_KEYS = {
    "text",
    "content",
    "body",
    "article",
    "answer",
    "definition",
    "description",
    "summary",
    "explanation",
    "example",
}
TITLE_KEYS = ("title", "headline", "name", "term", "question")
METADATA_KEYS = (
    "id",
    "schema_version",
    "collection",
    "category",
    "topic",
    "section",
    "difficulty",
    "content_type",
    "educational_only",
    "author",
    "date",
    "last_reviewed",
    "url",
    "source",
    "source_name",
    "source_url",
    "source_file",
    "tags",
    "aliases",
    "related_terms",
)


@dataclass
class RawDocument:
    text: str
    title: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    title: str
    metadata: dict[str, str | int | float | bool]


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _remove_trailing_commas(value: str) -> str:
    """Remove trailing JSON commas without changing content inside strings."""
    result: list[str] = []
    in_string = False
    escaped = False
    index = 0

    while index < len(value):
        character = value[index]
        if in_string:
            result.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue

        if character == '"':
            in_string = True
            result.append(character)
            index += 1
            continue

        if character == ",":
            next_index = index + 1
            while next_index < len(value) and value[next_index].isspace():
                next_index += 1
            if next_index < len(value) and value[next_index] in "]}":
                index += 1
                continue

        result.append(character)
        index += 1

    return "".join(result)


def _parse_json(value: str, source: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as original_exc:
        cleaned = _remove_trailing_commas(value)
        if cleaned == value:
            raise ValueError(
                f"Invalid JSON in {source}: {original_exc.msg} at line {original_exc.lineno}"
            ) from original_exc
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {source}: {exc.msg} at line {exc.lineno}"
            ) from exc


def _scalar_metadata(record: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {}
    for key in METADATA_KEYS:
        value = record.get(key)
        if isinstance(value, (str, int, float, bool)) and str(value).strip():
            metadata[key] = _clean(value) if isinstance(value, str) else value
        elif isinstance(value, list) and value and all(isinstance(item, (str, int, float, bool)) for item in value):
            metadata[key] = ", ".join(str(item) for item in value)
    return metadata


def _record_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    question = record.get("question")
    answer = record.get("answer")
    if isinstance(question, str) and isinstance(answer, str):
        parts.extend((f"Question: {_clean(question)}", f"Answer: {_clean(answer)}"))
        return "\n\n".join(parts)

    term = record.get("term")
    definition = record.get("definition")
    if isinstance(term, str) and isinstance(definition, str):
        parts.extend((f"Term: {_clean(term)}", f"Definition: {_clean(definition)}"))

        aliases = record.get("aliases")
        if isinstance(aliases, list):
            cleaned_aliases = [_clean(item) for item in aliases if isinstance(item, str) and _clean(item)]
            if cleaned_aliases:
                parts.append(f"Aliases: {', '.join(cleaned_aliases)}")

        example = record.get("example")
        if isinstance(example, str) and _clean(example):
            parts.append(f"Example: {_clean(example)}")

        related_terms = record.get("related_terms")
        if isinstance(related_terms, list):
            cleaned_terms = [
                _clean(item) for item in related_terms if isinstance(item, str) and _clean(item)
            ]
            if cleaned_terms:
                parts.append(f"Related terms: {', '.join(cleaned_terms)}")

        return "\n\n".join(parts)

    for key, value in record.items():
        if key.lower() in TEXT_KEYS and isinstance(value, str) and _clean(value):
            label = key.replace("_", " ").title()
            parts.append(f"{label}: {_clean(value)}")
    return "\n\n".join(parts)


def extract_documents(data: Any, source_name: str) -> list[RawDocument]:
    """Extract useful text records from nested JSON without assuming one schema."""
    documents: list[RawDocument] = []
    seen: set[str] = set()

    def add(text: str, title: str, path: str, metadata: dict[str, Any] | None = None) -> None:
        cleaned = _clean(text)
        if len(cleaned) < 20 or cleaned in seen:
            return
        seen.add(cleaned)
        safe_metadata: dict[str, str | int | float | bool] = {
            "source": source_name,
            "json_path": path,
        }
        for key, value in (metadata or {}).items():
            if isinstance(value, (str, int, float, bool)) and str(value).strip():
                safe_metadata[key] = _clean(value) if isinstance(value, str) else value
        documents.append(RawDocument(text=cleaned, title=_clean(title) or "Untitled", metadata=safe_metadata))

    def visit(node: Any, path: str = "$", inherited: dict[str, Any] | None = None) -> None:
        inherited = dict(inherited or {})
        if isinstance(node, dict):
            own_metadata = {**inherited, **_scalar_metadata(node)}

            # The Clarity common format keeps content records in a top-level
            # ``documents`` array. Ignore administrative siblings such as
            # ``source_policy`` so they do not become retrieval documents.
            if path == "$" and isinstance(node.get("documents"), list):
                for index, record in enumerate(node["documents"]):
                    visit(record, f"$.documents[{index}]", own_metadata)
                return

            title = next(
                (_clean(node[key]) for key in TITLE_KEYS if isinstance(node.get(key), str) and _clean(node[key])),
                path.rsplit(".", 1)[-1].replace("[", " ").replace("]", "").strip() or "Untitled",
            )
            text = _record_text(node)
            if text:
                add(text, title, path, own_metadata)
                return

            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    next_metadata = dict(own_metadata)
                    if isinstance(value, list) and "category" not in next_metadata:
                        next_metadata["category"] = key.replace("_", " ").title()
                    visit(value, f"{path}.{key}", next_metadata)
                elif isinstance(value, str) and len(_clean(value)) >= 40:
                    add(value, key.replace("_", " ").title(), f"{path}.{key}", own_metadata)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{path}[{index}]", inherited)
        elif isinstance(node, str) and len(_clean(node)) >= 40:
            add(node, path.rsplit(".", 1)[-1], path, inherited)

    visit(data)
    return documents


def load_json_documents(file_path: str | Path) -> list[RawDocument]:
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    if path.suffix.lower() not in {".json", ".jsonl", ".ndjson"}:
        raise ValueError("Expected a .json, .jsonl, or .ndjson file")

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        data = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                data.append(_parse_json(line, f"line {line_number}"))
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = _parse_json(handle.read(), path.name)

    documents = extract_documents(data, path.name)
    if not documents:
        raise ValueError(
            "No usable text was found. Include fields such as text, content, body, answer, definition, description, or summary."
        )
    return documents


def _split_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    if len(text) <= chunk_size:
        yield text
        return

    start = 0
    while start < len(text):
        target_end = min(len(text), start + chunk_size)
        end = target_end
        if target_end < len(text):
            boundary = text.rfind(" ", start + int(chunk_size * 0.65), target_end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def chunk_documents(
    documents: list[RawDocument], chunk_size: int = 1200, overlap: int = 180
) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size")
    chunks: list[Chunk] = []
    for document_index, document in enumerate(documents):
        for chunk_index, text in enumerate(_split_text(document.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    text=text,
                    title=document.title,
                    metadata={
                        **document.metadata,
                        "title": document.title,
                        "document_index": document_index,
                        "chunk_index": chunk_index,
                    },
                )
            )
    return chunks
