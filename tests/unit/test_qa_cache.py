import math

import pytest

from src.data.db import Database
from src.rag.qa_cache import QACache, normalize_question
from src.rag.vector_store import SearchResult

DIM = 8


def unit(*head):
    v = list(head) + [0.0] * (DIM - len(head))
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def at_cosine(c, axis=1):
    """A unit vector whose cosine with unit(1) is exactly c."""
    v = [0.0] * DIM
    v[0], v[axis] = c, math.sqrt(1 - c * c)
    return v


class VecEmbedder:
    """Maps exact question strings to chosen vectors so similarities are precise."""

    def __init__(self, table):
        self.table, self.calls = table, 0

    def __call__(self, text):
        self.calls += 1
        return self.table.get(text, unit(0, 0, 0, 0, 0, 0, 0, 1))


@pytest.fixture
def clock():
    return [10_000.0]


@pytest.fixture
def cache(tmp_path, clock):
    db = Database(tmp_path / "qa.db")
    db.init_schema()
    return QACache(db, exact_ttl=3600, semantic_ttl=7200, threshold=0.92, max_entries=50, clock=lambda: clock[0])


def hits(*titles):
    return [SearchResult(text=f"{t} passage", title=t, metadata={"source_name": "Investopedia", "source_url": f"https://ex.com/{t}",
                                                                 "category": "funds"}, score=0.7) for t in titles]


# ------------------------------------------------------------------------------ normalization
@pytest.mark.parametrize("a,b", [
    ("What is an ETF?", "what is an etf"), ("  What   is an ETF ?? ", "what is an etf"), ('"What is an ETF."', "what is an etf"),
    ("WHAT IS AN ETF!", "what is an etf"), ("Ｗhat is an ＥＴＦ?", "what is an etf"),
])
def test_normalize_question(a, b):
    assert normalize_question(a) == b


def test_normalize_keeps_meaningful_punctuation():
    assert normalize_question("What is a 401(k)?") == "what is a 401(k)"
    assert normalize_question("") == "" and normalize_question("?!.") == ""


# ---------------------------------------------------------------------------------- exact
def test_miss_then_store_then_exact_hit(cache):
    assert cache.lookup("What is an ETF?") is None
    cache.store("What is an ETF?", "An ETF is a basket.", hits("ETF", "Funds"))
    hit = cache.lookup("what is an ETF")  # different case and punctuation
    assert hit.kind == "exact" and hit.answer == "An ETF is a basket." and hit.question == "What is an ETF?"
    assert [r.title for r in hit.results] == ["ETF", "Funds"]
    assert hit.results[0].metadata["source_url"] == "https://ex.com/ETF" and hit.results[0].score == 0.7 and hit.similarity is None


def test_hit_reports_age(cache, clock):
    cache.store("What is a bond?", "A loan.", hits("Bond"))
    clock[0] += 600
    assert cache.lookup("What is a bond?").age_seconds == pytest.approx(600)


def test_exact_ttl_expiry(cache, clock):
    cache.store("What is a bond?", "A loan.", hits("Bond"))
    clock[0] += 3601
    assert cache.lookup("What is a bond?") is None


def test_scope_separates_categories(cache):
    cache.store("How are gains taxed?", "Tax answer", hits("Tax"), scope="tax")
    assert cache.lookup("How are gains taxed?", scope="") is None
    assert cache.lookup("How are gains taxed?", scope="bonds") is None
    assert cache.lookup("How are gains taxed?", scope="tax").answer == "Tax answer"
    assert cache.lookup("How are gains taxed?", scope=None) is None


def test_store_refuses_empty_answers_empty_sources_and_blank_questions(cache):
    cache.store("Q one?", "", hits("A"))
    cache.store("Q two?", "   ", hits("A"))
    cache.store("Q three?", "answer", [])
    cache.store("???", "answer", hits("A"))
    assert cache.stats()["entries"] == 0 and cache.lookup("   ") is None


def test_overwriting_a_question_replaces_the_answer_and_refreshes_age(cache, clock):
    cache.store("What is a bond?", "old", hits("Bond"))
    clock[0] += 3000
    cache.store("What is a bond?", "new", hits("Bond"))
    clock[0] += 1000  # 4000s after the first store, 1000s after the second
    hit = cache.lookup("What is a bond?")
    assert hit.answer == "new" and hit.age_seconds == pytest.approx(1000) and cache.stats()["entries"] == 1


def test_hit_count_is_tracked(cache):
    cache.store("What is a bond?", "A loan.", hits("Bond"))
    for _ in range(3):
        cache.lookup("what is a bond")
    with cache.db.connect() as conn:
        assert conn.execute("SELECT hit_count FROM qa_cache").fetchone()[0] == 3


# ------------------------------------------------------------------------- kb version invalidation
def test_bump_version_invalidates_everything(cache):
    cache.store("What is a bond?", "A loan.", hits("Bond"))
    before = cache.kb_version()
    cache.bump_version()
    assert cache.kb_version() != before and cache.lookup("What is a bond?") is None and cache.stats()["entries"] == 0


def test_stale_version_rows_are_ignored_even_if_left_in_the_table(cache):
    cache.store("What is a bond?", "A loan.", hits("Bond"))
    cache._meta.set("qa:kb_version", "another-process-changed-the-kb")  # e.g. `python -m src.rag.ingest` ran elsewhere
    assert cache.lookup("What is a bond?") is None


def test_version_is_shared_through_the_database(tmp_path, clock):
    db = Database(tmp_path / "shared.db")
    db.init_schema()
    a, b = QACache(db, clock=lambda: clock[0]), QACache(db, clock=lambda: clock[0])
    a.store("What is a bond?", "A loan.", hits("Bond"))
    assert b.lookup("What is a bond?").answer == "A loan."
    b.bump_version()
    assert a.lookup("What is a bond?") is None and a.kb_version() == b.kb_version()


def test_clear_returns_number_removed(cache):
    cache.store("Q one about bonds?", "a", hits("A"))
    cache.store("Q two about stocks?", "b", hits("B"))
    assert cache.clear() == 2 and cache.stats()["entries"] == 0


# ------------------------------------------------------------------------------ eviction / purge
def test_lru_eviction_keeps_recently_used(tmp_path, clock):
    db = Database(tmp_path / "e.db")
    db.init_schema()
    c = QACache(db, max_entries=3, clock=lambda: clock[0])
    for i in range(3):
        c.store(f"question number {i}", f"a{i}", hits("A"))
        clock[0] += 1
    c.lookup("question number 0")  # touch the oldest so it becomes most-recently-used
    clock[0] += 1
    c.store("question number 3", "a3", hits("A"))
    assert c.stats()["entries"] == 3
    assert c.lookup("question number 0") is not None  # kept: recently used
    assert c.lookup("question number 1") is None      # evicted: least recently used


def test_expired_entries_are_purged_on_store(cache, clock):
    cache.store("old question here", "a", hits("A"))
    clock[0] += 8000  # beyond both TTLs
    cache.store("fresh question here", "b", hits("B"))
    assert cache.stats()["entries"] == 1


# --------------------------------------------------------------------------------------- semantic
def test_semantic_hit_above_threshold_returns_similarity_and_original_question(cache):
    emb = VecEmbedder({"What is an ETF?": unit(1), "Can you explain what exchange traded funds are?": at_cosine(0.95)})
    cache.store("What is an ETF?", "An ETF is a basket.", hits("ETF"), embed=emb)
    hit = cache.lookup("Can you explain what exchange traded funds are?", embed=emb)
    assert hit.kind == "semantic" and hit.similarity == pytest.approx(0.95, abs=1e-3)
    assert hit.question == "What is an ETF?" and hit.answer == "An ETF is a basket."


def test_semantic_miss_below_threshold(cache):
    emb = VecEmbedder({"What is an ETF?": unit(1), "What is a mutual fund?": at_cosine(0.80)})
    cache.store("What is an ETF?", "a", hits("ETF"), embed=emb)
    assert cache.lookup("What is a mutual fund?", embed=emb) is None


@pytest.mark.parametrize("stored,asked", [
    ("What is a 401k?", "Explain what a 403b is"), ("Is $1,000 a good emergency fund?", "Is $10,000 a good emergency fund?"),
    ("Compare 2 funds", "Compare 3 funds"),
])
def test_semantic_match_requires_same_numbers(cache, stored, asked):
    emb = VecEmbedder({stored: unit(1), asked: at_cosine(0.99)})
    cache.store(stored, "a", hits("X"), embed=emb)
    assert cache.lookup(asked, embed=emb) is None


def test_same_numbers_in_different_formatting_still_match(cache):
    emb = VecEmbedder({"Is $10,000 enough to start?": unit(1), "Would $10000 be enough to start investing?": at_cosine(0.96)})
    cache.store("Is $10,000 enough to start?", "a", hits("X"), embed=emb)
    assert cache.lookup("Would $10000 be enough to start investing?", embed=emb).kind == "semantic"


def test_semantic_skips_number_mismatch_and_takes_next_best(cache):
    emb = VecEmbedder({"About the 401k plan": unit(1), "About the 403b plan": at_cosine(0.99),
                       "About the 401k retirement plan": at_cosine(0.94)})
    cache.store("About the 403b plan", "403b answer", hits("A"), embed=emb)
    cache.store("About the 401k retirement plan", "401k answer", hits("B"), embed=emb)
    hit = cache.lookup("About the 401k plan", embed=emb)
    assert hit.answer == "401k answer" and hit.similarity == pytest.approx(0.94, abs=1e-3)


def test_semantic_respects_scope_and_ttl(cache, clock):
    emb = VecEmbedder({"What is an ETF?": unit(1), "Explain ETFs please": at_cosine(0.96)})
    cache.store("What is an ETF?", "a", hits("ETF"), scope="funds", embed=emb)
    assert cache.lookup("Explain ETFs please", scope="", embed=emb) is None
    assert cache.lookup("Explain ETFs please", scope="funds", embed=emb) is not None
    clock[0] += 7300  # beyond semantic_ttl (7200)
    assert cache.lookup("Explain ETFs please", scope="funds", embed=emb) is None


def test_embedding_failures_never_break_lookup_or_store(cache):
    def broken(_):
        raise RuntimeError("embeddings down")

    cache.store("What is a bond?", "A loan.", hits("Bond"), embed=broken)  # stored without an embedding
    assert cache.lookup("what is a bond", embed=broken).kind == "exact"      # exact still works
    assert cache.lookup("Explain bonds to me", embed=broken) is None         # semantic degrades to a miss


def test_semantic_ignores_embeddings_from_other_models(cache):
    old = lambda text: [1.0, 0.0, 0.0]  # noqa: E731 - a 3-dim model
    new = VecEmbedder({"Explain ETFs please": unit(1)})
    cache.store("What is an ETF?", "a", hits("ETF"), embed=old)
    assert cache.lookup("Explain ETFs please", embed=new) is None  # dimension mismatch: skipped, no crash


def test_threshold_of_one_disables_semantic_matching(tmp_path, clock):
    db = Database(tmp_path / "t.db")
    db.init_schema()
    c = QACache(db, threshold=1.0, clock=lambda: clock[0])
    emb = VecEmbedder({"What is an ETF?": unit(1), "Explain ETFs please": at_cosine(0.99)})
    c.store("What is an ETF?", "a", hits("ETF"), embed=emb)
    assert c.lookup("Explain ETFs please", embed=emb) is None and c.lookup("what is an etf", embed=emb).kind == "exact"


def test_no_embed_function_means_exact_only(cache):
    cache.store("What is an ETF?", "a", hits("ETF"))
    assert cache.lookup("Explain ETFs please") is None


def test_exact_hit_does_not_call_the_embedder(cache):
    emb = VecEmbedder({})
    cache.store("What is an ETF?", "a", hits("ETF"), embed=emb)
    calls = emb.calls
    cache.lookup("what is an etf", embed=emb)
    assert emb.calls == calls  # exact match short-circuits: no embedding cost


# --------------------------------------------------------------------------------------- stats
def test_stats_and_hit_rate(cache):
    emb = VecEmbedder({"What is an ETF?": unit(1), "Explain ETFs please": at_cosine(0.96)})
    assert cache.stats()["hit_rate"] is None
    cache.lookup("What is an ETF?", embed=emb)                       # miss
    cache.store("What is an ETF?", "a", hits("ETF"), embed=emb)
    cache.lookup("what is an etf", embed=emb)                        # exact
    cache.lookup("Explain ETFs please", embed=emb)                   # semantic
    cache.lookup("Something unrelated entirely", embed=emb)          # miss
    s = cache.stats()
    assert (s["exact_hits"], s["semantic_hits"], s["misses"], s["stores"], s["entries"]) == (1, 1, 2, 1, 1)
    assert s["hit_rate"] == 0.5


# ------------------------------------------------------------------ guards found by live calibration
@pytest.mark.parametrize("stored,asked", [
    ("Is a 401k better than an IRA?", "Is an IRA better than a 401k?"),   # mirrored comparison, cos 0.94 in real embeddings
    ("Stocks or bonds for beginners?", "Bonds or stocks for beginners?"),
    ("Roth IRA vs traditional IRA", "Traditional IRA versus Roth IRA"),
    ("What is the difference between ETFs and mutual funds?", "Explain how ETFs and mutual funds differ, and the difference"),
])
def test_comparison_questions_are_never_matched_semantically(cache, stored, asked):
    emb = VecEmbedder({stored: unit(1), asked: at_cosine(0.99)})
    cache.store(stored, "a", hits("X"), embed=emb)
    assert cache.lookup(asked, embed=emb) is None


def test_comparison_questions_still_match_exactly(cache):
    q = "Is a 401k better than an IRA?"
    cache.store(q, "It depends.", hits("X"), embed=VecEmbedder({q: unit(1)}))
    assert cache.lookup("is a 401k better than an ira", embed=VecEmbedder({})).kind == "exact"


def test_a_stored_comparison_is_not_reused_for_a_plain_question(cache):
    emb = VecEmbedder({"Is a Roth IRA better than a savings account?": unit(1), "Tell me about Roth IRAs": at_cosine(0.99)})
    cache.store("Is a Roth IRA better than a savings account?", "a", hits("X"), embed=emb)
    assert cache.lookup("Tell me about Roth IRAs", embed=emb) is None


@pytest.mark.parametrize("stored,asked", [
    ("Is it wise to avoid index funds?", "Is it wise to buy index funds?"),
    ("Should I put money in crypto?", "Should I not put money in crypto?"),
    ("Can I invest without an advisor?", "Can I invest with an advisor?"),
])
def test_opposite_polarity_questions_do_not_share_answers(cache, stored, asked):
    emb = VecEmbedder({stored: unit(1), asked: at_cosine(0.99)})
    cache.store(stored, "a", hits("X"), embed=emb)
    assert cache.lookup(asked, embed=emb) is None


def test_matching_negation_words_still_allow_a_semantic_hit(cache):
    emb = VecEmbedder({"Why should I avoid penny stocks?": unit(1), "Reasons to avoid penny stocks, please": at_cosine(0.96)})
    cache.store("Why should I avoid penny stocks?", "a", hits("X"), embed=emb)
    assert cache.lookup("Reasons to avoid penny stocks, please", embed=emb).kind == "semantic"
