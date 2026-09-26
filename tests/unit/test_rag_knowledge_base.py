import pytest

from src.core.config import RagConfig
from src.rag import knowledge_base as kbmod
from src.rag.categories import CATEGORY_GROUPS, group_of, resolve_categories
from src.rag.knowledge_base import KnowledgeBase, build_knowledge_base, to_citations
from src.rag.loaders import chunk_documents, load_json_documents
from src.rag.qdrant_store import QdrantHybridStore
from src.rag.vector_store import FinanceVectorStore, SearchResult
from tests.unit._fakes import BagEmbedder, CORPUS


class FakeStore:
    def __init__(self, name, results=None, count=1, fail_search=False, fail_ingest=False, fail_count=False):
        self.name, self._results, self._count = name, results or [], count
        self.fail_search, self.fail_ingest, self.fail_count = fail_search, fail_ingest, fail_count
        self.search_calls, self.ingested, self.was_reset = [], [], False

    def count(self):
        if self.fail_count:
            raise RuntimeError("count broke")
        return self._count

    def search(self, query, top_k, *, category=None, mode="hybrid"):
        self.search_calls.append((query, top_k, category, mode))
        if self.fail_search:
            raise RuntimeError("search broke")
        return [SearchResult(r.text, r.title, dict(r.metadata), r.score) for r in self._results]

    def ingest(self, chunks, reset=False):
        if self.fail_ingest:
            raise RuntimeError("ingest broke")
        self.ingested.append((len(chunks), reset))
        return len(chunks)

    def categories(self):
        return {"a": 1}

    def reset(self):
        self.was_reset = True


def hit(title="T", score=0.7, **meta):
    return SearchResult(text=f"{title} text", title=title, metadata=dict(meta), score=score)


# -------------------------------------------------------------------------------------- search
def test_primary_result_is_used_and_labelled():
    kb = KnowledgeBase(FakeStore("qdrant", [hit()]), FakeStore("faiss", [hit("FB")]))
    out = kb.search("q", 3)
    assert [r.title for r in out] == ["T"] and out[0].metadata["backend"] == "qdrant"
    assert kb.fallback.search_calls == []


def test_primary_error_falls_back_and_labels_backend():
    kb = KnowledgeBase(FakeStore("qdrant", fail_search=True), FakeStore("faiss", [hit("FB")]))
    out = kb.search("q", 3)
    assert [r.title for r in out] == ["FB"] and out[0].metadata["backend"] == "faiss"


def test_empty_primary_index_falls_back():
    kb = KnowledgeBase(FakeStore("qdrant", [], count=0), FakeStore("faiss", [hit("FB")]))
    assert kb.search("q", 3)[0].metadata["backend"] == "faiss"


def test_no_match_in_populated_primary_does_not_fall_back():
    kb = KnowledgeBase(FakeStore("qdrant", [], count=200), FakeStore("faiss", [hit("FB")]))
    assert kb.search("q", 3) == [] and kb.fallback.search_calls == []


def test_both_failing_returns_empty_not_an_exception():
    kb = KnowledgeBase(FakeStore("qdrant", fail_search=True), FakeStore("faiss", fail_search=True))
    assert kb.search("q", 3) == []


def test_no_fallback_configured():
    kb = KnowledgeBase(FakeStore("qdrant", fail_search=True))
    assert kb.search("q", 3) == []
    assert KnowledgeBase(FakeStore("qdrant", [hit()])).search("q", 3)[0].title == "T"


def test_category_group_is_resolved_and_lists_pass_through():
    kb = KnowledgeBase(FakeStore("qdrant", [hit()]))
    kb.search("q", 3, category="tax")
    assert kb.primary.search_calls[-1][2] == list(CATEGORY_GROUPS["tax"])
    kb.search("q", 3, category=["bonds"])
    assert kb.primary.search_calls[-1][2] == ["bonds"]
    kb.search("q", 3, category="stocks")
    assert kb.primary.search_calls[-1][2] == ["stocks"]
    kb.search("q", 3)
    assert kb.primary.search_calls[-1][2] is None


# ------------------------------------------------------------------------------ ingest / status
def test_ingest_writes_both_stores_and_reset_is_forwarded():
    kb = KnowledgeBase(FakeStore("qdrant"), FakeStore("faiss"))
    assert kb.ingest(CORPUS, reset=True) == len(CORPUS)
    assert kb.primary.ingested == [(len(CORPUS), True)] and kb.fallback.ingested == [(len(CORPUS), True)]


def test_fallback_ingest_failure_is_tolerated_but_primary_failure_is_not():
    ok = KnowledgeBase(FakeStore("qdrant"), FakeStore("faiss", fail_ingest=True))
    assert ok.ingest(CORPUS) == len(CORPUS)
    with pytest.raises(RuntimeError):
        KnowledgeBase(FakeStore("qdrant", fail_ingest=True), FakeStore("faiss")).ingest(CORPUS)


def test_count_prefers_primary_then_fallback():
    assert KnowledgeBase(FakeStore("q", count=5), FakeStore("f", count=9)).count() == 5
    assert KnowledgeBase(FakeStore("q", count=0), FakeStore("f", count=9)).count() == 9
    assert KnowledgeBase(FakeStore("q", fail_count=True), FakeStore("f", count=9)).count() == 9
    assert KnowledgeBase(FakeStore("q", count=0)).count() == 0


def test_status_reports_each_store_and_survives_failures():
    st = KnowledgeBase(FakeStore("qdrant", count=4), FakeStore("faiss", fail_count=True)).status()
    assert st["qdrant"] == {"chunks": 4, "ok": True} and st["faiss"]["ok"] is False and st["faiss"]["error"] == "RuntimeError"


def test_reset_and_categories():
    kb = KnowledgeBase(FakeStore("qdrant"), FakeStore("faiss"))
    kb.reset()
    assert kb.primary.was_reset and kb.fallback.was_reset and kb.categories() == {"a": 1}

    class NoCats(FakeStore):
        def categories(self):
            raise RuntimeError("x")

    assert KnowledgeBase(NoCats("q"), FakeStore("f")).categories() == {"a": 1}
    assert KnowledgeBase(NoCats("q")).categories() == {}


# ---------------------------------------------------------------------------------- citations
def test_citations_prefer_publisher_name_and_url_over_file_name():
    r1 = hit("Diversification", 0.71234, source="kb.json", source_name="Investopedia", source_url="https://ex.com/d", category="portfolio_management")
    r2 = hit("Legacy", 0.5, source="upload.json")
    c1, c2 = to_citations([r1, r2])
    assert (c1.index, c1.source, c1.url, c1.category, c1.score) == (1, "Investopedia", "https://ex.com/d", "portfolio_management", 0.712)
    assert (c2.index, c2.source, c2.url) == (2, "upload.json", None)  # falls back to the file name, no invented URL


def test_citation_excerpt_is_truncated():
    long = SearchResult(text="x" * 500, title="L", metadata={}, score=0.5)
    c = to_citations([long])[0]
    assert len(c.excerpt) == 261 and c.excerpt.endswith("…")
    assert to_citations([]) == []


# ------------------------------------------------------------------------------------- factory
def test_factory_builds_configured_backends(tmp_path, monkeypatch):
    monkeypatch.setattr(kbmod.settings.__class__, "qdrant_path", tmp_path / "q", raising=False)
    for name, value in (("qdrant_path", tmp_path / "q"), ("faiss_path", tmp_path / "f")):
        object.__setattr__(kbmod.settings, name, value)
    emb = BagEmbedder()
    both = build_knowledge_base(emb, RagConfig(backend="qdrant", fallback="faiss"))
    assert [s.name for s in both.stores] == ["qdrant", "faiss"]
    assert isinstance(both.primary, QdrantHybridStore) and isinstance(both.fallback, FinanceVectorStore)
    both.primary.close()
    only = build_knowledge_base(emb, RagConfig(backend="faiss", fallback=""))
    assert [s.name for s in only.stores] == ["faiss"]
    same = build_knowledge_base(emb, RagConfig(backend="faiss", fallback="faiss"))
    assert same.fallback is None
    with pytest.raises(ValueError, match="Unknown retriever backend"):
        build_knowledge_base(emb, RagConfig(backend="pinecone", fallback=""))


def test_embeddings_are_shared_between_primary_and_fallback_on_ingest(tmp_path):
    for name, value in (("qdrant_path", tmp_path / "q"), ("faiss_path", tmp_path / "f")):
        object.__setattr__(kbmod.settings, name, value)
    raw = BagEmbedder()
    kb = build_knowledge_base(raw, RagConfig(backend="qdrant", fallback="faiss"))
    kb.ingest(CORPUS, reset=True)
    assert raw.texts == len(CORPUS)  # each chunk embedded ONCE even though two indexes were written
    assert kb.search("bond maturity interest", 1)[0].title == "Bond Basics"
    kb.primary.close()


# ------------------------------------------------------------------------------- category groups
def test_resolve_and_group_of():
    assert resolve_categories(None) is None and resolve_categories("") is None
    assert resolve_categories(" TAX ") == list(CATEGORY_GROUPS["tax"])
    assert resolve_categories("Bonds") == ["bonds"] and resolve_categories("custom_cat") == ["custom_cat"]
    assert "tax" in group_of("tax_efficiency") and group_of("nonexistent") == []


def test_every_category_in_the_real_knowledge_base_belongs_to_a_group():
    docs = load_json_documents("src/data/kb/clarity_financial_education_common.json")
    categories = {c.metadata["category"] for c in chunk_documents(docs)}
    orphans = {c for c in categories if not group_of(c)}
    assert not orphans, f"KB categories with no group (add them to CATEGORY_GROUPS): {sorted(orphans)}"
    assert len(categories) >= 40


def test_real_kb_has_titles_categories_and_publishers_for_every_chunk():
    chunks = chunk_documents(load_json_documents("src/data/kb/clarity_financial_education_common.json"))
    assert len(chunks) >= 100  # rubric "excellent" tier: 100+ articles
    assert all(c.title and c.title != "Untitled" for c in chunks)
    assert all(c.metadata.get("category") and c.metadata.get("source_name") for c in chunks)


# ------------------------------------------------------------------------------ sourced-article preference
def _r(title, score, url=None, **meta):
    m = {"source_url": url} if url else {}
    return SearchResult(f"{title} text", title, {**m, **meta}, score)


def test_sourced_article_wins_a_near_tie_but_not_a_clear_loss():
    from src.rag.knowledge_base import has_source_link, prefer_sourced

    unsourced, sourced = _r("Placeholder", 0.70), _r("Investopedia", 0.64, "https://ex.com/a")
    assert [r.title for r in prefer_sourced([unsourced, sourced], 0.08)] == ["Investopedia", "Placeholder"]
    far = _r("Far", 0.40, "https://ex.com/b")
    assert [r.title for r in prefer_sourced([unsourced, far], 0.08)] == ["Placeholder", "Far"]
    assert has_source_link(SearchResult("t", "x", {"url": "https://ex.com"}, 0.5)) and not has_source_link(_r("x", 0.5))
    assert (unsourced.score, sourced.score) == (0.70, 0.64)  # displayed scores are untouched


def test_prefer_sourced_is_stable_and_can_be_disabled():
    from src.rag.knowledge_base import prefer_sourced

    items = [_r("a", 0.5), _r("b", 0.5), _r("c", 0.5)]
    assert [r.title for r in prefer_sourced(items, 0.08)] == ["a", "b", "c"]
    swapped = [_r("low", 0.3, "https://ex.com"), _r("high", 0.6)]
    assert prefer_sourced(swapped, 0) == swapped and prefer_sourced(swapped[:1], 0.08) == swapped[:1]


def test_knowledge_base_overfetches_then_returns_the_top_k_with_sourced_first():
    results = [_r("Placeholder", 0.70), _r("Investopedia", 0.64, "https://ex.com/a"), _r("Other", 0.50), _r("Extra", 0.45)]
    store = FakeStore("qdrant", results)
    out = KnowledgeBase(store).search("q", 2)
    assert store.search_calls[0][1] == 6  # top_k * overfetch (3)
    assert [r.title for r in out] == ["Investopedia", "Placeholder"]
