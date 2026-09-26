import pytest

from src.rag.qdrant_store import QdrantHybridStore
from tests.unit._fakes import BagEmbedder, CORPUS, mk_chunk


@pytest.fixture
def store():
    s = QdrantHybridStore(":memory:", "kb", BagEmbedder(), candidates=10)
    s.ingest(CORPUS)
    return s


def titles(results):
    return [r.title for r in results]


def test_empty_store_returns_nothing_and_counts_zero():
    s = QdrantHybridStore(":memory:", "kb", BagEmbedder())
    assert s.count() == 0 and s.search("anything", 3) == [] and s.categories() == {}


def test_ingest_counts_and_is_idempotent(store):
    assert store.count() == len(CORPUS)
    store.ingest(CORPUS)  # same chunks -> same ids -> no duplicates
    assert store.count() == len(CORPUS)


def test_ingest_empty_and_reset():
    s = QdrantHybridStore(":memory:", "kb", BagEmbedder())
    assert s.ingest([]) == 0
    s.ingest(CORPUS)
    s.ingest(CORPUS[:2], reset=True)
    assert s.count() == 2
    s.reset()
    assert s.count() == 0


def test_batching_writes_everything():
    s = QdrantHybridStore(":memory:", "kb", BagEmbedder())
    many = [mk_chunk(f"Doc {i}", f"unique term{i} content about investing", idx=i) for i in range(150)]
    assert s.ingest(many, batch_size=40) == 150 and s.count() == 150


def test_dimension_change_is_rejected_with_clear_message():
    s = QdrantHybridStore(":memory:", "kb", BagEmbedder())
    s.ingest(CORPUS)

    class Other:
        embed_model = "other"

        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    s.embedder = Other()
    with pytest.raises(ValueError, match="dimension changed"):
        s.ingest([mk_chunk("New", "brand new content", idx=99)])


@pytest.mark.parametrize("mode", ["hybrid", "dense", "sparse"])
def test_every_mode_finds_the_obvious_match(store, mode):
    assert titles(store.search("Sharpe ratio risk return", 1, mode=mode)) == ["Sharpe Ratio"]


def test_results_carry_cosine_score_source_metadata_and_retrieval_label(store):
    hit = store.search("Roth IRA retirement account", 1)[0]
    assert hit.title == "Roth IRA" and 0.0 <= hit.score <= 1.0 and hit.score > 0.3
    assert hit.metadata["source_name"] == "SEC Investor.gov" and hit.metadata["category"] == "tax_education"
    assert "text" not in hit.metadata and hit.text.startswith("A Roth IRA")
    assert hit.metadata["retrieval"] in {"dense", "sparse", "dense+sparse"} and hit.metadata["rrf_score"] > 0


def test_document_found_by_both_lists_gets_a_fusion_bonus(store):
    hit = store.search("capital gains tax taxable account sell", 1)[0]
    assert hit.title == "Capital Gains" and hit.metadata["retrieval"] == "dense+sparse"


def test_keyword_only_match_is_found_by_sparse_even_with_weak_dense(store):
    # 'maturity' appears in exactly one chunk; sparse mode must surface it
    assert titles(store.search("maturity", 1, mode="sparse")) == ["Bond Basics"]


def test_sparse_mode_with_only_stopwords_returns_nothing(store):
    assert store.search("the of and", 3, mode="sparse") == []


@pytest.mark.parametrize("cat,expected", [
    ("bonds", {"Bond Basics"}),
    (["tax_education", "taxes_and_investing"], {"Roth IRA", "Capital Gains"}),
])
def test_category_filter(store, cat, expected):
    results = store.search("investment account", 5, category=cat)
    assert results and {r.title for r in results} <= expected


def test_category_filter_with_no_match_is_empty(store):
    assert store.search("investing", 5, category="no_such_category") == []


def test_top_k_limits_and_invalid_args(store):
    assert len(store.search("investment", 2)) == 2
    assert store.search("investment", 0) == [] and store.search("investment", -1) == []
    assert store.search("investment", 3, mode="bogus") == []


def test_categories_counts(store):
    cats = store.categories()
    assert cats["bonds"] == 1 and cats["tax_education"] == 1 and sum(cats.values()) == len(CORPUS)


def test_query_embedding_is_requested_once_per_search(store):
    calls_before = store.embedder.calls
    store.search("diversification", 3)
    assert store.embedder.calls == calls_before + 1


def test_on_disk_persistence_across_reopen(tmp_path):
    path = tmp_path / "q"
    first = QdrantHybridStore(path, "kb", BagEmbedder())
    first.ingest(CORPUS)
    first.close()
    second = QdrantHybridStore(path, "kb", BagEmbedder())
    assert second.count() == len(CORPUS)
    assert second.search("bond interest maturity", 1)[0].title == "Bond Basics"
    second.close()
    second.close()  # closing twice must be safe


def test_aliases_boost_keyword_matching():
    s = QdrantHybridStore(":memory:", "kb", BagEmbedder())
    s.ingest([mk_chunk("Emergency Fund", "Cash for surprises.", aliases="rainy day fund", idx=0),
              mk_chunk("Index Funds", "Funds that track a market index.", idx=1)])
    assert s.search("rainy day", 1, mode="sparse")[0].title == "Emergency Fund"


def test_unicode_and_long_text_are_handled():
    s = QdrantHybridStore(":memory:", "kb", BagEmbedder())
    s.ingest([mk_chunk("Café Économie", "Résumé sur l'épargne — naïve investors save early. " * 30, idx=0)])
    assert s.search("naïve investors save", 1)[0].title == "Café Économie"
