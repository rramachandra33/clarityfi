"""KnowledgeBase: one retrieval facade over the primary (Qdrant hybrid) and fallback (FAISS) stores.

- ingest() writes to every configured store (embeddings are computed once and shared).
- search() uses the primary; if it raises or its index is empty, the fallback answers and the
  result metadata records which backend was used (`backend`), so degradation is visible in traces.
- to_citations() converts hits into the shared Citation model for source attribution in answers.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence

from src.core.config import RagConfig, get_config, settings
from src.core.models import Citation
from src.rag.categories import resolve_categories
from src.rag.embedding import CachingEmbedder, Embedder
from src.rag.loaders import Chunk
from src.rag.vector_store import FinanceVectorStore, SearchResult

logger = logging.getLogger(__name__)
EXCERPT_CHARS = 260


class Retriever(Protocol):
    name: str

    def count(self) -> int: ...
    def ingest(self, chunks: list[Chunk], reset: bool = False) -> int: ...
    def search(self, query: str, top_k: int, *, category: Any = None, mode: str = ...) -> list[SearchResult]: ...
    def categories(self) -> dict[str, int]: ...
    def reset(self) -> None: ...


class KnowledgeBase:
    def __init__(self, primary: Retriever, fallback: Retriever | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.on_change: list[Any] = []  # callbacks run after ingest/reset (e.g. invalidate the answer cache)

    def _changed(self) -> None:
        for callback in self.on_change:
            try:
                callback()
            except Exception:
                logger.exception("Knowledge-base change callback failed")

    @property
    def stores(self) -> list[Retriever]:
        return [self.primary] + ([self.fallback] if self.fallback else [])

    # -- counts / metadata -----------------------------------------------------------------
    def count(self) -> int:
        """Chunks in the primary index (falls back to the fallback's count if the primary is unusable)."""
        try:
            n = self.primary.count()
            if n or not self.fallback:
                return n
        except Exception:
            logger.exception("Primary index count failed")
        return self.fallback.count() if self.fallback else 0

    def categories(self) -> dict[str, int]:
        for store in self.stores:
            try:
                cats = store.categories()
                if cats:
                    return cats
            except Exception:
                logger.exception("%s categories() failed", store.name)
        return {}

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for store in self.stores:
            try:
                out[store.name] = {"chunks": store.count(), "ok": True}
            except Exception as exc:
                out[store.name] = {"chunks": 0, "ok": False, "error": type(exc).__name__}
        return out

    # -- ingest -----------------------------------------------------------------------------
    def ingest(self, chunks: list[Chunk], reset: bool = False) -> int:
        """Write to all stores. Primary failure aborts; a fallback failure is logged (search still works)."""
        written = self.primary.ingest(chunks, reset=reset)
        if self.fallback:
            try:
                self.fallback.ingest(chunks, reset=reset)
            except Exception:
                logger.exception("Fallback index ingest failed; primary is up to date")
        self._changed()
        return written

    def reset(self) -> None:
        for store in self.stores:
            store.reset()
        self._changed()

    # -- search ------------------------------------------------------------------------------
    def search(self, query: str, top_k: int, *, category: Sequence[str] | str | None = None,
               mode: str = "hybrid") -> list[SearchResult]:
        """`category` may be a group ("tax"), a raw category, or a list of raw categories."""
        cats = resolve_categories(category) if isinstance(category, str) else (list(category) if category else None)
        cfg = get_config().rag
        fetch_k = top_k * max(1, cfg.overfetch) if cfg.sourced_bonus else top_k
        results: list[SearchResult] | None = None
        try:
            results = self.primary.search(query, fetch_k, category=cats, mode=mode)
            backend = self.primary.name
        except Exception:
            logger.exception("%s search failed; using fallback", self.primary.name)
            backend = None
        if (not results) and self.fallback is not None:
            try:
                if backend is None or self.primary.count() == 0:
                    results = self.fallback.search(query, fetch_k, category=cats)
                    backend = self.fallback.name
            except Exception:
                logger.exception("Fallback search failed")
                results = results or []
        results = results or []
        for r in results:
            r.metadata.setdefault("backend", backend or self.primary.name)
        return prefer_sourced(results, cfg.sourced_bonus)[:top_k]


def has_source_link(result: SearchResult) -> bool:
    return bool(result.metadata.get("source_url") or result.metadata.get("url"))


def prefer_sourced(results: list[SearchResult], bonus: float) -> list[SearchResult]:
    """Order results by relevance plus a small boost for articles that carry a real source link.

    Some articles have only a placeholder source and no link; when a sourced article is nearly as relevant, users
    (and the answer's citations) are better served by the one they can verify. The boost only reorders: each result's
    own `score` is unchanged, so relevance gates and displayed scores stay honest. Sorting is stable.
    """
    if not bonus or len(results) < 2:
        return results
    return sorted(results, key=lambda r: r.score + (bonus if has_source_link(r) else 0.0), reverse=True)


def to_citations(results: list[SearchResult]) -> list[Citation]:
    """Numbered citations ([1], [2], ...) with source name/link for transparent attribution."""
    citations = []
    for index, r in enumerate(results, start=1):
        meta = r.metadata
        source = meta.get("source_name") or meta.get("source")
        url = meta.get("source_url") or meta.get("url")
        citations.append(Citation(
            index=index, title=r.title, source=str(source) if source else None, url=str(url) if url else None,
            category=str(meta["category"]) if meta.get("category") else None,
            excerpt=r.text[:EXCERPT_CHARS] + ("…" if len(r.text) > EXCERPT_CHARS else ""),
            score=round(r.score, 3)))
    return citations


def build_knowledge_base(embedder: Embedder, config: RagConfig | None = None) -> KnowledgeBase:
    """Assemble primary/fallback stores from config.yaml. Embeddings are shared via one CachingEmbedder."""
    cfg = config or get_config().rag
    shared = embedder if isinstance(embedder, CachingEmbedder) else CachingEmbedder(embedder)

    def make(kind: str) -> Retriever:
        if kind == "qdrant":
            from src.rag.qdrant_store import QdrantHybridStore

            return QdrantHybridStore(settings.qdrant_path, settings.qdrant_collection, shared,
                                     url=settings.qdrant_url or None, candidates=cfg.candidates, rrf_k=cfg.rrf_k)
        if kind == "faiss":
            return FinanceVectorStore(settings.faiss_path, settings.faiss_collection, shared)
        raise ValueError(f"Unknown retriever backend '{kind}' (use 'qdrant' or 'faiss')")

    primary = make(cfg.backend)
    fallback = make(cfg.fallback) if cfg.fallback and cfg.fallback != cfg.backend else None
    return KnowledgeBase(primary, fallback)
