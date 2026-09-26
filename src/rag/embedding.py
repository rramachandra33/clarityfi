"""Embedding wrapper: memoizes by text so ingest embeds each chunk once for BOTH indexes, and
repeated queries do not re-call the API."""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Protocol


class Embedder(Protocol):
    embed_model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class CachingEmbedder:
    def __init__(self, inner: Embedder, max_entries: int = 4096) -> None:
        self.inner = inner
        self.embed_model = inner.embed_model
        self._max = max_entries
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.api_calls = 0  # texts actually sent to the provider (for cost visibility in tests/logs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            vectors = self.inner.embed(missing)
            if len(vectors) != len(missing):
                raise ValueError("Embedding provider returned a different number of vectors than requested")
            with self._lock:
                self.api_calls += len(missing)
                for text, vector in zip(missing, vectors):
                    self._cache[text] = vector
                while len(self._cache) > self._max:
                    self._cache.popitem(last=False)
        with self._lock:
            out = []
            for text in texts:
                self._cache.move_to_end(text)
                out.append(self._cache[text])
            return out
