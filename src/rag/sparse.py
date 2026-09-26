"""Keyword (sparse) encoding for hybrid search: BM25-style term weights hashed to integer ids.

Qdrant applies the IDF factor server-side (Modifier.IDF), so documents only need a saturating
term-frequency weight and queries only need the set of terms. No model download, fully offline.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOPWORDS = frozenset("""a an and are as at be been but by can could did do does for from had has have how i if in into is it its
me my of on or our should so than that the their them then there these they this to us was we were what when where which who
whom why will with would you your about above after again against all am any because before being below between both down
during each few further he her here hers him his itself just more most no nor not now off once only other out over own same
she some such too under until up very via""".split())
_HASH_SPACE = 2**31 - 1
K1 = 1.2  # term-frequency saturation


def stem(token: str) -> str:
    """Very light stemming so 'funds'/'fund' and 'investing'/'invest' match."""
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("es") and token[-3] in "sxz":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in STOPWORDS or len(raw) < 2 and not raw.isdigit():
            continue
        tokens.append(stem(raw))
        if "-" in raw:  # also index the parts: 'tax-exempt' -> 'tax', 'exempt'
            tokens.extend(stem(p) for p in raw.split("-") if p and p not in STOPWORDS and len(p) > 1)
    return tokens


def token_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) % _HASH_SPACE  # stable across processes (unlike hash())


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


class SparseEncoder:
    """encode_document -> saturated TF weights; encode_query -> unit weight per distinct term."""

    def encode_document(self, text: str) -> SparseVec:
        counts: dict[int, int] = {}
        for token in tokenize(text):
            idx = token_id(token)
            counts[idx] = counts.get(idx, 0) + 1
        items = sorted(counts.items())
        return SparseVec([i for i, _ in items], [round(c * (K1 + 1) / (c + K1), 4) for _, c in items])

    def encode_query(self, text: str) -> SparseVec:
        ids = sorted({token_id(t) for t in tokenize(text)})
        return SparseVec(ids, [1.0] * len(ids))
