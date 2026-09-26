import threading

import pytest

from src.rag.embedding import CachingEmbedder
from src.rag.sparse import SparseEncoder, stem, token_id, tokenize
from tests.unit._fakes import BagEmbedder


# ---------------------------------------------------------------------------------- tokenizer
@pytest.mark.parametrize("word,expected", [
    ("funds", "fund"), ("bonds", "bond"), ("stocks", "stock"), ("investing", "invest"), ("companies", "company"),
    ("taxes", "tax"), ("boxes", "box"), ("class", "class"), ("bus", "bus"), ("analysis", "analysis"),
    ("etf's", "etf"), ("sing", "sing"), ("is", "is"),
])
def test_stem_cases(word, expected):
    assert stem(word) == expected


def test_tokenize_lowercases_drops_stopwords_and_stems():
    assert tokenize("What are the Funds and Bonds?") == ["fund", "bond"]
    assert tokenize("How does an IRA work") == ["ira", "work"]


def test_tokenize_keeps_numbers_and_splits_hyphenated_terms():
    toks = tokenize("tax-exempt 401k and 60/40")
    assert "401k" in toks and "60" in toks and "40" in toks
    assert "tax-exempt" in toks and "tax" in toks and "exempt" in toks


def test_tokenize_empty_and_symbols():
    assert tokenize("") == [] and tokenize("!!! ???") == [] and tokenize("the and of") == []


def test_token_id_is_stable_and_bounded():
    assert token_id("diversification") == token_id("diversification")
    import zlib
    assert token_id("etf") == zlib.crc32(b"etf") % (2**31 - 1)  # pinned: ids must never change between releases
    assert 0 <= token_id("anything at all") < 2**31 - 1
    assert token_id("fund") != token_id("bond")


def test_document_encoding_saturates_term_frequency():
    enc = SparseEncoder()
    once = enc.encode_document("tax")
    many = enc.encode_document("tax " * 50)
    assert once.values[0] == pytest.approx(1.0) and 1.0 < many.values[0] < 2.2 + 1e-9  # saturates below k1+1
    assert len(many.indices) == 1


def test_document_encoding_sorted_unique_indices():
    v = SparseEncoder().encode_document("fund bond fund stock bond fund")
    assert v.indices == sorted(set(v.indices)) and len(v.indices) == 3 and len(v.values) == 3


def test_query_encoding_is_unit_weights_and_deduped():
    q = SparseEncoder().encode_query("funds funds and the bond")
    assert len(q.indices) == 2 and q.values == [1.0, 1.0]
    empty = SparseEncoder().encode_query("the of and")
    assert empty.indices == [] and empty.values == []


def test_query_and_document_share_ids_for_matching_terms():
    enc = SparseEncoder()
    assert set(enc.encode_query("Roth IRA").indices) <= set(enc.encode_document("A Roth IRA is a retirement account").indices)


# ---------------------------------------------------------------------------------- embedder
def test_caching_embedder_calls_provider_only_for_new_texts():
    inner = BagEmbedder()
    emb = CachingEmbedder(inner)
    a = emb.embed(["alpha", "beta"])
    b = emb.embed(["beta", "gamma", "alpha"])
    assert inner.texts == 3 and emb.api_calls == 3  # alpha, beta, gamma once each
    assert b[0] == a[1] and b[2] == a[0]  # order preserved
    assert emb.embed_model == inner.embed_model


def test_caching_embedder_dedupes_within_a_batch():
    inner = BagEmbedder()
    CachingEmbedder(inner).embed(["same", "same", "same"])
    assert inner.texts == 1


def test_caching_embedder_evicts_least_recently_used():
    inner = BagEmbedder()
    emb = CachingEmbedder(inner, max_entries=2)
    emb.embed(["a", "b"])
    emb.embed(["a"])          # a becomes most recent
    emb.embed(["c"])          # evicts b
    before = inner.texts
    emb.embed(["a"])
    assert inner.texts == before  # a still cached
    emb.embed(["b"])
    assert inner.texts == before + 1  # b was evicted


def test_caching_embedder_detects_short_provider_response():
    class Broken:
        embed_model = "x"

        def embed(self, texts):
            return [[1.0]]

    with pytest.raises(ValueError, match="different number"):
        CachingEmbedder(Broken()).embed(["a", "b"])


def test_caching_embedder_empty_input_and_thread_safety():
    inner = BagEmbedder()
    emb = CachingEmbedder(inner)
    assert emb.embed([]) == []
    results = []

    def worker(i):
        results.append(emb.embed([f"text {i % 5}"])[0])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(results) == 30 and emb.api_calls >= 5
