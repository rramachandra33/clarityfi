from src.rag.loaders import Chunk
from src.rag.vector_store import FinanceVectorStore


class FakeOllama:
    embed_model = "test-embedding"

    def embed(self, texts):
        return [
            [float("etf" in text.lower()), float("bond" in text.lower()), 0.1]
            for text in texts
        ]


def test_persistent_ingest_and_similarity_search(tmp_path):
    store = FinanceVectorStore(tmp_path / "chroma", "finance_test", FakeOllama())
    chunks = [
        Chunk(
            text="An ETF holds a basket of securities.",
            title="ETF basics",
            metadata={"source": "test.json", "json_path": "$[0]", "chunk_index": 0, "title": "ETF basics"},
        ),
        Chunk(
            text="A bond is a debt security.",
            title="Bond basics",
            metadata={"source": "test.json", "json_path": "$[1]", "chunk_index": 0, "title": "Bond basics"},
        ),
    ]

    assert store.ingest(chunks, reset=True) == 2
    assert store.count() == 2
    results = store.search("How does an ETF work?", top_k=1)
    assert results[0].title == "ETF basics"
    assert results[0].score > 0.9


# ------------------------------------------------------------------------------- robustness / errors
import json  # noqa: E402

import pytest  # noqa: E402


def _chunk(text, i=0, **meta):
    return Chunk(text=text, title=text[:10], metadata={"source": "t.json", "json_path": f"$[{i}]", "chunk_index": 0, "title": text[:10], **meta})


def _store(tmp_path, embedder=None):
    return FinanceVectorStore(tmp_path / "f", "coll", embedder or FakeOllama())


def test_reload_from_disk_and_category_counts_and_upsert(tmp_path):
    s = _store(tmp_path)
    s.ingest([_chunk("An ETF basket", 0, category="funds"), _chunk("A bond loan", 1, category="bonds")], reset=True)
    again = _store(tmp_path)  # loads persisted index + records
    assert again.count() == 2 and again.categories() == {"bonds": 1, "funds": 1}
    assert again.ingest([_chunk("An ETF basket", 0, category="funds")]) == 1 and again.count() == 2  # same id: upsert, no duplicate
    assert [r.title for r in again.search("bond", 1, category="bonds")] == ["A bond loa"]
    assert again.search("etf", 0) == [] and again._search_raw("etf", 0) == [] and again.ingest([]) == 0


def test_bad_arguments_and_embedding_failures(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(ValueError, match="batch_size"):
        s.ingest([_chunk("x")], batch_size=0)
    for bad in ([[float("nan"), 1.0, 0.0]], [[0.0, 0.0, 0.0]], [[1.0, 2.0], [3.0]], "nope"):
        e = FakeOllama()
        e.embed = lambda texts, bad=bad: bad
        with pytest.raises(ValueError):
            _store(tmp_path / "x", e).ingest([_chunk("x")])
    assert s.search("anything", 3) == []  # empty index


def test_dimension_change_is_rejected(tmp_path):
    s = _store(tmp_path)
    s.ingest([_chunk("An ETF")], reset=True)
    e = FakeOllama()
    e.embed = lambda texts: [[1.0, 0.0] for _ in texts]
    s.openai = e
    with pytest.raises(ValueError, match="dimension changed"):
        s.ingest([_chunk("Another", 1)])
    with pytest.raises(ValueError, match="dimension"):
        s.search("q", 1)


def test_corrupt_or_mismatched_collections_are_refused(tmp_path):
    s = _store(tmp_path)
    s.ingest([_chunk("An ETF")], reset=True)
    records = s._records_path
    good = json.loads(records.read_text())

    def rewrite(**changes):
        records.write_text(json.dumps({**good, **changes}))

    rewrite(collection_name="other")
    with pytest.raises(ValueError, match="collection name"):
        _store(tmp_path)
    rewrite(embedding_model="different-model")
    with pytest.raises(ValueError, match="embedding model"):
        _store(tmp_path)
    rewrite(dimension="bad")
    with pytest.raises(ValueError, match="invalid"):
        _store(tmp_path)
    rewrite(records=[])
    with pytest.raises(ValueError, match="inconsistent"):
        _store(tmp_path)
    records.write_text("{broken")
    with pytest.raises(ValueError, match="metadata"):
        _store(tmp_path)
    records.write_text(json.dumps(good))
    s._index_path.write_bytes(b"not an index")
    with pytest.raises(ValueError, match="FAISS index"):
        _store(tmp_path)
    records.unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        _store(tmp_path)
