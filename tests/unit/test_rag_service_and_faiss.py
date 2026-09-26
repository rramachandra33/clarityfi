import pytest
from fastapi.testclient import TestClient

from src.core.models import AgentName, AgentResult, Citation
from src.rag.vector_store import FinanceVectorStore
from src.web_app.api import main
from tests.unit._agent_fakes import FakeAgent, make_workflow
from tests.unit._fakes import BagEmbedder, CORPUS, mk_chunk


# ------------------------------------------------------------------------------ FAISS filter
@pytest.fixture
def faiss_store(tmp_path):
    s = FinanceVectorStore(tmp_path / "f", "kb", BagEmbedder())
    s.ingest(CORPUS)
    return s


def test_faiss_category_filter_string_and_list(faiss_store):
    assert {x.title for x in faiss_store.search("investment tax account", 5, category="taxes_and_investing")} == {"Capital Gains"}
    got = faiss_store.search("investment account tax", 5, category=["tax_education", "bonds"])
    assert {x.title for x in got} <= {"Roth IRA", "Bond Basics"} and got


def test_faiss_filter_overfetches_so_filtered_results_are_still_found(tmp_path):
    s = FinanceVectorStore(tmp_path / "f2", "kb", BagEmbedder())
    noise = [mk_chunk(f"Noise {i}", f"market investing stock fund portfolio {i}", "stocks", idx=i) for i in range(60)]
    target = mk_chunk("Rare Topic", "market investing stock fund portfolio xylophone", "special", idx=100)
    s.ingest(noise + [target])
    assert [x.title for x in s.search("market investing stock fund portfolio", 1, category="special")] == ["Rare Topic"]


def test_faiss_no_filter_no_match_and_categories(faiss_store):
    assert len(faiss_store.search("investment", 3)) == 3
    assert faiss_store.search("investment", 3, category="nope") == []
    assert faiss_store.search("investment", 0) == []
    assert faiss_store.categories()["bonds"] == 1 and faiss_store.name == "faiss"


# --------------------------------------------------------------------------- API attribution
def test_chat_sources_use_publisher_and_url_not_file_name(monkeypatch):
    agent = FakeAgent("finance_qa", "Diversification reduces single-holding risk. [1]", citations=[
        Citation(index=1, title="Diversification", source="Investopedia", url="https://ex.com/div",
                 category="portfolio_management", excerpt="Diversification spreads risk", score=0.71),
        Citation(index=2, title="Emergency Fund", source="Starter content", excerpt="Starter passage", score=0.5)])
    monkeypatch.setattr(main, "workflow", make_workflow([agent]))
    body = TestClient(main.app).post("/api/chat", json={"user_id": "learner-204", "question": "What is diversification?"}).json()
    first, second = body["sources"]
    assert (first["source"], first["url"], first["category"], first["index"]) == ("Investopedia", "https://ex.com/div", "portfolio_management", 1)
    assert second["source"] == "Starter content" and second["url"] is None  # no invented link
    assert first["score"] == pytest.approx(0.71)
