"""Finance Q&A answer cache: exact match first, then semantic (embedding) match.

Safety rules (enforced by the caller, `FinanceQAAgent`, and documented here):
- Only SELF-CONTAINED questions are cached: follow-ups depend on the conversation.
- Nothing is cached when PII was redacted from the question (it was personal).
- Refusals ("not enough information") are never cached.
- Questions are stored REDACTED (post-Presidio); answers come from the knowledge base only.
- Entries are tied to a knowledge-base version: ingesting new content invalidates every entry,
  even when the ingest runs as a separate process (the version lives in the database).
- A semantic match must also agree on the numbers in the question ("401k" vs "403b", "$1,000" vs
  "$10,000") and on negation words ("avoid"/"not"), because embeddings treat those as nearly identical.
- Comparison questions ("A vs B", "better than", "or") are matched EXACTLY only: the mirrored question
  ("B vs A") scores ~0.94 similar in real embeddings but asks something different.

Every hit is reported to the user (see `build_qa_cache_notice`); nothing is served silently.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.data.cache import PersistentCache
from src.data.db import Database
from src.rag.vector_store import SearchResult

VERSION_KEY = "qa:kb_version"
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*[a-z]?")
# Embeddings treat mirrored comparisons ("401k better than IRA?" vs "IRA better than 401k?") as ~0.94 similar,
# so comparison questions are only ever matched EXACTLY, never semantically.
_COMPARATIVE = re.compile(r"\b(vs\.?|versus|than|compare[sd]?|comparison|difference|differences|different|better|worse|or)\b")
_NEGATIONS = frozenset({"not", "no", "never", "without", "avoid", "except", "don't", "dont", "shouldn't", "isn't", "aren't",
                        "won't", "can't", "cannot", "unless", "instead", "against"})


def normalize_question(question: str) -> str:
    """Case/spacing/trailing-punctuation-insensitive form used for the exact-match key."""
    text = unicodedata.normalize("NFKC", question).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\"'`?!.,;:")


def _numbers(question: str) -> frozenset[str]:
    return frozenset(m.replace(",", "") for m in _NUMBER.findall(question.lower()))


def _is_comparative(question: str) -> bool:
    return bool(_COMPARATIVE.search(question.lower()))


def _negations(question: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z']+", question.lower()) if w in _NEGATIONS)


def _unit(vector: list[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


@dataclass
class CacheHit:
    kind: str                       # "exact" | "semantic"
    answer: str
    results: list[SearchResult]
    age_seconds: float
    question: str                   # the stored (redacted) question that produced this answer
    similarity: float | None = None


class QACache:
    def __init__(
        self,
        db: Database,
        *,
        exact_ttl: float = 86400,
        semantic_ttl: float = 86400,
        threshold: float = 0.92,
        max_entries: int = 500,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db = db
        self.exact_ttl, self.semantic_ttl = exact_ttl, semantic_ttl
        self.threshold, self.max_entries = threshold, max_entries
        self._clock = clock
        self._meta = PersistentCache(db, clock=clock)
        self._lock = threading.Lock()
        self.counters = {"exact_hits": 0, "semantic_hits": 0, "misses": 0, "stores": 0}

    # -- knowledge-base versioning -------------------------------------------------------------
    def kb_version(self) -> str:
        found = self._meta.get_stale(VERSION_KEY, 10**12)
        return str(found[0]) if found else "0"

    def bump_version(self) -> None:
        """Call whenever knowledge-base content changes; drops every cached answer."""
        self._meta.set(VERSION_KEY, uuid.uuid4().hex)
        self.clear()

    def clear(self) -> int:
        with self.db.connect() as conn:
            return conn.execute("DELETE FROM qa_cache").rowcount

    # -- lookup ---------------------------------------------------------------------------------
    @staticmethod
    def _key(scope: str, norm: str) -> str:
        return hashlib.sha256(f"{scope}\x1f{norm}".encode("utf-8")).hexdigest()

    @staticmethod
    def _results(sources_json: str) -> list[SearchResult]:
        return [SearchResult(text=s["text"], title=s["title"], metadata=s.get("metadata") or {}, score=float(s["score"]))
                for s in json.loads(sources_json)]

    def lookup(self, question: str, scope: str = "", embed: Callable[[str], list[float]] | None = None) -> CacheHit | None:
        """Exact match, else semantic match; None on a miss. `embed` computes the question embedding."""
        scope = scope or ""
        version, now, norm = self.kb_version(), self._clock(), normalize_question(question)
        if not norm:
            return None
        key = self._key(scope, norm)

        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM qa_cache WHERE key_hash = ? AND kb_version = ?", (key, version)).fetchone()
            if row and now - row["created_at"] <= self.exact_ttl:
                conn.execute("UPDATE qa_cache SET hit_count = hit_count + 1, last_used_at = ? WHERE key_hash = ?", (now, key))
                self._count("exact_hits")
                return CacheHit("exact", row["answer"], self._results(row["sources_json"]), now - row["created_at"], row["question"])

        if embed is not None and self.threshold < 1.0 and not _is_comparative(question):
            try:
                query = _unit(embed(question.strip()))
            except Exception:  # embedding outage must never break answering: treat as a miss
                query = None
            if query is not None and query.size:
                hit = self._semantic(query, question, scope, version, now)
                if hit:
                    return hit
        self._count("misses")
        return None

    def _semantic(self, query: np.ndarray, question: str, scope: str, version: str, now: float) -> CacheHit | None:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT key_hash, question, answer, sources_json, embedding, created_at FROM qa_cache "
                "WHERE scope = ? AND kb_version = ? AND embedding IS NOT NULL AND created_at >= ? "
                "ORDER BY last_used_at DESC LIMIT ?",
                (scope, version, now - self.semantic_ttl, self.max_entries),
            ).fetchall()
            rows = [r for r in rows if len(r["embedding"]) == query.size * 4]  # skip other embedding models
            if not rows:
                return None
            matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            sims = matrix @ query
            wanted = _numbers(question)
            for idx in np.argsort(-sims):
                similarity = float(sims[idx])
                if similarity < self.threshold:
                    break
                row = rows[int(idx)]
                stored = row["question"]
                if _numbers(stored) != wanted or _negations(stored) != _negations(question) or _is_comparative(stored):
                    continue  # different figures, opposite polarity, or a comparison: not the same question
                conn.execute("UPDATE qa_cache SET hit_count = hit_count + 1, last_used_at = ? WHERE key_hash = ?",
                             (now, row["key_hash"]))
                self._count("semantic_hits")
                return CacheHit("semantic", row["answer"], self._results(row["sources_json"]),
                                now - row["created_at"], row["question"], round(similarity, 4))
        return None

    # -- store ----------------------------------------------------------------------------------
    def store(self, question: str, answer: str, results: list[SearchResult], scope: str = "",
              embed: Callable[[str], list[float]] | None = None) -> None:
        scope, norm, now = scope or "", normalize_question(question), self._clock()
        if not norm or not answer.strip() or not results:
            return  # never cache empty answers or answers with no supporting sources
        blob = None
        if embed is not None:
            try:
                blob = _unit(embed(question.strip())).astype(np.float32).tobytes()
            except Exception:
                blob = None
        sources = json.dumps([{"text": r.text, "title": r.title, "metadata": r.metadata, "score": r.score} for r in results],
                             default=str)
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO qa_cache (key_hash, scope, kb_version, question_norm, question, answer, sources_json, embedding, "
                "created_at, last_used_at, hit_count) VALUES (?,?,?,?,?,?,?,?,?,?,0) "
                "ON CONFLICT(key_hash) DO UPDATE SET kb_version=excluded.kb_version, question=excluded.question, "
                "answer=excluded.answer, sources_json=excluded.sources_json, embedding=excluded.embedding, "
                "created_at=excluded.created_at, last_used_at=excluded.last_used_at, hit_count=0",
                (self._key(scope, norm), scope, self.kb_version(), norm, question.strip(), answer, sources, blob, now, now),
            )
            conn.execute("DELETE FROM qa_cache WHERE created_at < ?", (now - max(self.exact_ttl, self.semantic_ttl),))
            conn.execute(
                "DELETE FROM qa_cache WHERE key_hash IN (SELECT key_hash FROM qa_cache ORDER BY last_used_at DESC LIMIT -1 OFFSET ?)",
                (self.max_entries,),
            )
        self._count("stores")

    # -- observability --------------------------------------------------------------------------
    def _count(self, name: str) -> None:
        with self._lock:
            self.counters[name] += 1

    def stats(self) -> dict[str, Any]:
        with self.db.connect() as conn:
            entries = conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
        with self._lock:
            c = dict(self.counters)
        lookups = c["exact_hits"] + c["semantic_hits"] + c["misses"]
        return {"entries": entries, **c, "hit_rate": round((c["exact_hits"] + c["semantic_hits"]) / lookups, 3) if lookups else None}
