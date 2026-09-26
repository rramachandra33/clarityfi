"""Qdrant hybrid store: dense (OpenAI embeddings) + sparse (BM25-style keywords), fused with RRF.

Runs embedded (no server) from a local folder, or against a Qdrant server when a URL is given.
We run the dense and sparse queries ourselves and fuse in Python so that:
  * ranking uses Reciprocal Rank Fusion (robust across the two score scales), and
  * every result keeps its dense COSINE similarity, which is the score used for "is this actually
    relevant?" gating (RRF scores are rank-based and say nothing about absolute relevance).
"""
from __future__ import annotations

import atexit
import hashlib
import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from qdrant_client import QdrantClient, models as qm

from src.rag.embedding import Embedder
from src.rag.loaders import Chunk
from src.rag.sparse import SparseEncoder
from src.rag.vector_store import SearchResult

logger = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("6f1c2f0e-9b1c-4d7e-8a53-0c1a4f7d2b11")
_PAYLOAD_META_SKIP = {"json_path", "document_index", "chunk_index"}


def _point_id(chunk: Chunk) -> str:
    raw = f"{chunk.metadata.get('source')}|{chunk.metadata.get('json_path')}|{chunk.metadata.get('chunk_index')}|{chunk.text}"
    return str(uuid.uuid5(_NAMESPACE, hashlib.sha256(raw.encode("utf-8")).hexdigest()))


def _sparse_text(chunk: Chunk) -> str:
    """Keyword index text: title and aliases are repeated so exact-term queries rank them highly."""
    meta = chunk.metadata
    extras = " ".join(str(meta.get(k, "")) for k in ("aliases", "related_terms", "category"))
    return f"{chunk.title} {chunk.title} {extras} {chunk.text}"


class QdrantHybridStore:
    name = "qdrant"

    def __init__(
        self,
        path: str | Path | None,
        collection: str,
        embedder: Embedder,
        *,
        url: str | None = None,
        candidates: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.collection = collection
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.encoder = SparseEncoder()
        self._lock = threading.RLock()
        if url:
            self.client = QdrantClient(url=url)
        elif path in (None, ":memory:"):
            self.client = QdrantClient(":memory:")
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
            atexit.register(self.close)  # release the local folder lock cleanly at interpreter exit

    # ---------------------------------------------------------------------------- collection
    def _exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def _dimension(self) -> int | None:
        if not self._exists():
            return None
        vectors = self.client.get_collection(self.collection).config.params.vectors
        return vectors["dense"].size if isinstance(vectors, dict) else None

    def _create(self, dimension: int) -> None:
        self.client.create_collection(
            self.collection,
            vectors_config={"dense": qm.VectorParams(size=dimension, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )

    def count(self) -> int:
        with self._lock:
            return self.client.count(self.collection, exact=True).count if self._exists() else 0

    def reset(self) -> None:
        with self._lock:
            if self._exists():
                self.client.delete_collection(self.collection)

    def categories(self) -> dict[str, int]:
        """category -> number of chunks (for UI filters and category-group validation)."""
        counts: dict[str, int] = {}
        with self._lock:
            if not self._exists():
                return counts
            offset = None
            while True:
                points, offset = self.client.scroll(self.collection, limit=256, offset=offset, with_payload=["category"])
                for p in points:
                    cat = (p.payload or {}).get("category")
                    if cat:
                        counts[cat] = counts.get(cat, 0) + 1
                if offset is None:
                    break
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------------------------- ingest
    def ingest(self, chunks: list[Chunk], reset: bool = False, batch_size: int = 64) -> int:
        if not chunks:
            return 0
        with self._lock:
            if reset:
                self.reset()
            written = 0
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                vectors = self.embedder.embed([c.text for c in batch])
                dimension = len(vectors[0])
                existing = self._dimension()
                if existing is None:
                    self._create(dimension)
                elif existing != dimension:
                    raise ValueError(
                        f"Embedding dimension changed from {existing} to {dimension}. "
                        "Reset the collection before changing embedding models."
                    )
                points = []
                for chunk, dense in zip(batch, vectors):
                    sparse = self.encoder.encode_document(_sparse_text(chunk))
                    payload = {k: v for k, v in chunk.metadata.items() if k not in _PAYLOAD_META_SKIP}
                    payload.update({"text": chunk.text, "title": chunk.title})
                    points.append(qm.PointStruct(
                        id=_point_id(chunk),
                        vector={"dense": dense, "sparse": qm.SparseVector(indices=sparse.indices, values=sparse.values)},
                        payload=payload,
                    ))
                self.client.upsert(self.collection, points)  # idempotent: same chunk -> same id
                written += len(points)
            return written

    # ------------------------------------------------------------------------------- search
    @staticmethod
    def _filter(category: Sequence[str] | str | None) -> qm.Filter | None:
        if not category:
            return None
        values = [category] if isinstance(category, str) else list(category)
        return qm.Filter(must=[qm.FieldCondition(key="category", match=qm.MatchAny(any=values))])

    def search(self, query: str, top_k: int, *, category: Sequence[str] | str | None = None,
               mode: str = "hybrid") -> list[SearchResult]:
        """mode: 'hybrid' (default), 'dense', or 'sparse'."""
        if top_k <= 0 or mode not in {"hybrid", "dense", "sparse"}:
            return []
        with self._lock:
            if not self._exists() or self.count() == 0:
                return []
            flt = self._filter(category)
            n = max(self.candidates, top_k)

            dense_vec = None
            dense_hits: list[Any] = []
            if mode in {"hybrid", "dense"}:
                dense_vec = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
                dense_hits = self.client.query_points(
                    self.collection, query=dense_vec.tolist(), using="dense", limit=n, query_filter=flt,
                    with_payload=True, with_vectors=["dense"]).points
            sparse_hits: list[Any] = []
            if mode in {"hybrid", "sparse"}:
                sq = self.encoder.encode_query(query)
                if sq.indices:
                    sparse_hits = self.client.query_points(
                        self.collection, query=qm.SparseVector(indices=sq.indices, values=sq.values), using="sparse",
                        limit=n, query_filter=flt, with_payload=True, with_vectors=["dense"]).points

        # Reciprocal Rank Fusion
        fused: dict[str, dict[str, Any]] = {}
        for label, hits in (("dense", dense_hits), ("sparse", sparse_hits)):
            for rank, hit in enumerate(hits, start=1):
                entry = fused.setdefault(str(hit.id), {"hit": hit, "rrf": 0.0, "via": []})
                entry["rrf"] += 1.0 / (self.rrf_k + rank)
                entry["via"].append(label)

        if dense_vec is None:  # sparse-only mode still needs a query vector for the relevance score
            dense_vec = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        qn = dense_vec / (np.linalg.norm(dense_vec) or 1.0)

        results = []
        for entry in sorted(fused.values(), key=lambda e: -e["rrf"])[:top_k]:
            hit = entry["hit"]
            payload = dict(hit.payload or {})
            stored = np.asarray((hit.vector or {}).get("dense", []), dtype=np.float32)
            cosine = float(np.dot(qn, stored / (np.linalg.norm(stored) or 1.0))) if stored.size else 0.0
            text, title = str(payload.pop("text", "")), str(payload.get("title") or "Untitled")
            payload["retrieval"] = "+".join(entry["via"])
            payload["rrf_score"] = round(entry["rrf"], 5)
            results.append(SearchResult(text=text, title=title, metadata=payload, score=max(0.0, min(1.0, cosine))))
        return results

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # already closed / interpreter shutting down
            pass
