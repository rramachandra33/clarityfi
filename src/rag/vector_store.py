from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from src.rag.loaders import Chunk

class OpenAIClient(Protocol):
    embed_model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...

@dataclass
class SearchResult:
    text: str
    title: str
    metadata: dict[str, Any]
    score: float


class FinanceVectorStore:
    name = "faiss"
    """Persistent FAISS vector store using cosine similarity.

    FAISS stores normalized vectors in ``index.faiss``. Document text,
    metadata, and the embedding model name are stored in ``records.json`` in a
    collection-specific directory.
    """

    def __init__(self, path: Path, collection_name: str, openai: OpenAIClient) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.openai = openai
        self._lock = threading.RLock()

        collection_key = hashlib.sha256(collection_name.encode("utf-8")).hexdigest()[:16]
        self._collection_path = self.path / f"faiss_{collection_key}"
        self._collection_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self._collection_path / "index.faiss"
        self._records_path = self._collection_path / "records.json"

        self._index: Any | None = None
        self._records: list[dict[str, Any]] = []
        self._dimension: int | None = None
        self._load()

    def _load(self) -> None:
        index_exists = self._index_path.exists()
        records_exist = self._records_path.exists()

        if index_exists != records_exist:
            raise ValueError(
                f"Incomplete FAISS collection at '{self._collection_path}'. "
                "Reset the collection and ingest the documents again."
            )
        if not index_exists:
            return

        try:
            with self._records_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read FAISS metadata at '{self._records_path}'."
            ) from exc

        if payload.get("collection_name") != self.collection_name:
            raise ValueError("The persisted FAISS collection name does not match the configuration.")

        model = payload.get("embedding_model")
        if model and model != self.openai.embed_model:
            raise ValueError(
                f"Collection uses embedding model '{model}', but configuration uses "
                f"'{self.openai.embed_model}'. Reset the collection before changing "
                "embedding models."
            )

        records = payload.get("records")
        dimension = payload.get("dimension")
        if not isinstance(records, list) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("The persisted FAISS metadata is invalid.")

        try:
            index = faiss.read_index(str(self._index_path))
        except Exception as exc:
            raise ValueError(f"Could not read FAISS index at '{self._index_path}'.") from exc

        if index.d != dimension or index.ntotal != len(records):
            raise ValueError(
                "The FAISS index and document metadata are inconsistent. "
                "Reset the collection and ingest the documents again."
            )

        self._index = index
        self._records = records
        self._dimension = dimension

    @staticmethod
    def _validated_vectors(embeddings: Any, expected_count: int) -> np.ndarray:
        try:
            vectors = np.asarray(embeddings, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("The embedding provider returned invalid vectors.") from exc

        if vectors.ndim != 2 or vectors.shape[0] != expected_count or vectors.shape[1] == 0:
            raise ValueError(
                f"Expected {expected_count} embedding vectors with a consistent dimension."
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding vectors must contain only finite numbers.")

        norms = np.linalg.norm(vectors, axis=1)
        if any(math.isclose(float(norm), 0.0, abs_tol=1e-12) for norm in norms):
            raise ValueError("The embedding provider returned a zero-length vector.")

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    def _existing_vectors(self) -> np.ndarray:
        if self._index is None or self._index.ntotal == 0:
            dimension = self._dimension or 0
            return np.empty((0, dimension), dtype=np.float32)

        vectors = np.empty((self._index.ntotal, self._index.d), dtype=np.float32)
        self._index.reconstruct_n(0, self._index.ntotal, vectors)
        return vectors

    def _persist(self) -> None:
        if self._index is None or self._dimension is None:
            return

        index_tmp = self._index_path.with_name(f"{self._index_path.name}.tmp")
        records_tmp = self._records_path.with_name(f"{self._records_path.name}.tmp")
        payload = {
            "collection_name": self.collection_name,
            "embedding_model": self.openai.embed_model,
            "dimension": self._dimension,
            "records": self._records,
        }

        try:
            faiss.write_index(self._index, str(index_tmp))
            with records_tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(index_tmp, self._index_path)
            os.replace(records_tmp, self._records_path)
        finally:
            index_tmp.unlink(missing_ok=True)
            records_tmp.unlink(missing_ok=True)

    def reset(self) -> None:
        with self._lock:
            self._index = None
            self._records = []
            self._dimension = None
            self._index_path.unlink(missing_ok=True)
            self._records_path.unlink(missing_ok=True)

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def ingest(self, chunks: list[Chunk], reset: bool = False, batch_size: int = 32) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        with self._lock:
            if reset:
                self.reset()
            if not chunks:
                return 0

            existing_vectors = self._existing_vectors()
            dimension = self._dimension
            records_by_id = {
                str(record["id"]): (record, existing_vectors[position])
                for position, record in enumerate(self._records)
            }

            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                documents = [chunk.text for chunk in batch]
                vectors = self._validated_vectors(
                    self.openai.embed(documents), expected_count=len(batch)
                )

                if dimension is None:
                    dimension = int(vectors.shape[1])
                elif vectors.shape[1] != dimension:
                    raise ValueError(
                        f"Embedding dimension changed from {dimension} to "
                        f"{vectors.shape[1]}. Reset the collection before changing "
                        "embedding models."
                    )

                for chunk, vector in zip(batch, vectors):
                    record_id = hashlib.sha256(
                        (
                            f"{chunk.metadata.get('source')}|"
                            f"{chunk.metadata.get('json_path')}|"
                            f"{chunk.metadata.get('chunk_index')}|{chunk.text}"
                        ).encode("utf-8")
                    ).hexdigest()[:32]
                    records_by_id[record_id] = (
                        {
                            "id": record_id,
                            "text": chunk.text,
                            "metadata": dict(chunk.metadata),
                        },
                        vector,
                    )

            self._records = [record for record, _ in records_by_id.values()]
            all_vectors = np.ascontiguousarray(
                np.vstack([vector for _, vector in records_by_id.values()]),
                dtype=np.float32,
            )
            self._dimension = dimension
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(all_vectors)
            self._persist()
            return len(chunks)

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._lock:
            for record in self._records:
                cat = (record.get("metadata") or {}).get("category")
                if cat:
                    counts[str(cat)] = counts.get(str(cat), 0) + 1
        return dict(sorted(counts.items()))

    def search(self, query: str, top_k: int, *, category: Any = None, mode: str = "dense") -> list[SearchResult]:
        """Dense-only search. `category` (str or list) filters results; `mode` is accepted for interface parity."""
        if top_k <= 0:
            return []
        wanted = None if not category else ({category} if isinstance(category, str) else set(category))
        if wanted:
            # FAISS has no metadata filter. The flat index is brute-force anyway, so rank EVERY vector,
            # then filter: a fixed over-fetch could miss a rarely-matching category entirely.
            results = self._search_raw(query, max(self.count(), top_k))
            return [r for r in results if str(r.metadata.get("category")) in wanted][:top_k]
        return self._search_raw(query, top_k)

    def _search_raw(self, query: str, top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            return []

        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []

            query_vector = self._validated_vectors(
                self.openai.embed([query]), expected_count=1
            )
            if query_vector.shape[1] != self._index.d:
                raise ValueError(
                    f"Query embedding has dimension {query_vector.shape[1]}, but the "
                    f"FAISS index uses dimension {self._index.d}."
                )

            scores, positions = self._index.search(
                query_vector, min(top_k, self._index.ntotal)
            )
            matches: list[SearchResult] = []
            for score, position in zip(scores[0], positions[0]):
                if position < 0:
                    continue
                record = self._records[int(position)]
                metadata = record.get("metadata") or {}
                matches.append(
                    SearchResult(
                        text=str(record.get("text") or ""),
                        title=str(metadata.get("title") or "Untitled"),
                        metadata=metadata,
                        score=max(0.0, min(1.0, float(score))),
                    )
                )
            return matches
