"""Load JSON finance content into the knowledge base (Qdrant hybrid index + FAISS fallback).

    python -m src.rag.ingest --reset                      # rebuild from the bundled knowledge base
    python -m src.rag.ingest --file my.json               # add/update from another file

Stop the API server first when using embedded Qdrant (its local folder allows one process).
"""
from __future__ import annotations

import argparse
import sys

from src.core.config import PROJECT_ROOT, get_config, settings
from src.data.db import Database
from src.core.openai_client import OpenAIClient, OpenAIError
from src.rag.embedding import CachingEmbedder
from src.rag.knowledge_base import build_knowledge_base
from src.rag.loaders import chunk_documents, load_json_documents
from src.rag.qa_cache import QACache

DEFAULT_KB = PROJECT_ROOT / "src/data/kb/clarity_financial_education_common.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Load JSON finance content into the knowledge base.")
    parser.add_argument("--file", default=str(DEFAULT_KB), help="Path to a .json, .jsonl, or .ndjson file (default: bundled KB)")
    parser.add_argument("--reset", action="store_true", help="Clear the indexes before ingesting")
    args = parser.parse_args()

    client = OpenAIClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        chat_model=settings.openai_chat_model,
        embed_model=settings.openai_embed_model,
    )
    embedder = CachingEmbedder(client)
    kb = build_knowledge_base(embedder)
    db = Database(settings.app_db_path)
    db.init_schema()
    kb.on_change.append(QACache(db).bump_version)  # cached answers are stale once content changes
    try:
        documents = load_json_documents(args.file)
        chunks = chunk_documents(documents)
        count = kb.ingest(chunks, reset=args.reset)
    except (ValueError, FileNotFoundError, OpenAIError) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(documents)} documents as {count} chunks ({embedder.api_calls} texts embedded).")
    for name, info in kb.status().items():
        print(f"  {name}: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
