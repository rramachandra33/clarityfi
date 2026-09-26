"""Health/ingest/glossary/stats routes, the legacy OpenAI embeddings client, glossary + loader edge cases, and the CLIs."""
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from src.core.openai_client import OpenAIClient, OpenAIError
from src.data.glossary_store import GlossaryAlreadyInitializedError, GlossaryStore, load_glossary_file
from src.rag import loaders
from src.web_app.api import main


# ------------------------------------------------------------------------------------------ routes
class FakeStore:
    def __init__(self, count=5):
        self._count, self.ingested = count, []

    def count(self):
        if self._count is None:
            raise RuntimeError("index unavailable")
        return self._count

    def ingest(self, chunks, reset=False):
        self.ingested.append((chunks, reset))
        return len(chunks)


@pytest.fixture
def api(tmp_path, monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "glossary_store", GlossaryStore(tmp_path / "g.db"))
    monkeypatch.setattr(main, "glossary_initialization_error", None)
    monkeypatch.setattr(main.openai_client, "is_available", lambda: True)
    with TestClient(main.app) as client:
        client.store = store
        yield client


def test_health_states(api, monkeypatch):
    assert api.get("/api/health").json()["status"] == "ready"
    monkeypatch.setattr(main.openai_client, "is_available", lambda: False)
    assert api.get("/api/health").json()["status"] == "needs_setup"
    api.store._count = None  # index errors are contained
    body = api.get("/api/health").json()
    assert body["knowledge_chunks"] == 0 and body["status"] == "needs_setup"


def test_stats(api):
    body = api.get("/api/stats").json()
    assert body["knowledge_chunks"] == 5 and "qa_cache" in body


def test_ingest_file_paths(api):
    doc = json.dumps([{"title": "Bonds", "content": "A bond is a loan to a company or government. " * 3, "category": "bonds"}])
    ok = api.post("/api/ingest/file", files={"file": ("kb.json", doc, "application/json")}, data={"reset": "true"})
    assert ok.status_code == 200 and ok.json()["documents"] == 1 and api.store.ingested[0][1] is True
    assert api.post("/api/ingest/file", files={"file": ("kb.txt", "x", "text/plain")}).status_code == 400
    assert api.post("/api/ingest/file", files={"file": ("kb.json", "{not json", "application/json")}).status_code == 400


def test_ingest_file_too_large(api, monkeypatch):
    monkeypatch.setattr(main, "settings", SimpleNamespace(**{**main.settings.__dict__, "max_upload_mb": 0}))
    assert api.post("/api/ingest/file", files={"file": ("kb.json", "[]", "application/json")}).status_code == 413


def test_ingest_embedding_failure_is_503(api, monkeypatch):
    def boom(chunks, reset=False):
        raise OpenAIError("embedding service down")
    monkeypatch.setattr(api.store, "ingest", boom)
    doc = json.dumps([{"title": "Bonds", "content": "A bond is a loan. " * 5}])
    assert api.post("/api/ingest/file", files={"file": ("kb.json", doc, "application/json")}).status_code == 503


def test_glossary_lifecycle(api):
    empty = api.get("/api/glossary").json()
    assert empty["initialized"] is False and empty["items"] == []
    good = json.dumps([{"term": "ETF", "definition": "A fund that trades like a stock."}, {"term": "Bond", "definition": "A loan."}])
    assert api.post("/api/glossary/initialize", files={"file": ("g.json", good, "application/json")}).json() == {"initialized": True, "count": 2}
    assert [i["term"] for i in api.get("/api/glossary?q=etf").json()["items"]] == ["ETF"]
    again = api.post("/api/glossary/initialize", files={"file": ("g.json", good, "application/json")})
    assert again.status_code == 409


def test_glossary_initialize_rejects_bad_files(api):
    assert api.post("/api/glossary/initialize", files={"file": ("g.txt", "x", "text/plain")}).status_code == 400
    bad = api.post("/api/glossary/initialize", files={"file": ("g.json", json.dumps([{"term": "x"}]), "application/json")})
    assert bad.status_code == 400
    assert api.get("/api/glossary").json()["initialization_error"]


# --------------------------------------------------------------------------- legacy OpenAI client
REAL_CLIENT = httpx.Client


def client_with(handler, monkeypatch):
    real = REAL_CLIENT
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **{k: v for k, v in kw.items() if k != "transport"}))
    return OpenAIClient("k", "https://api.example.com/v1/", "chat", "embed")


def embedding_response(vectors):
    return httpx.Response(200, json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]})


def test_embed_success_orders_by_index_and_empty_input(monkeypatch):
    c = client_with(lambda req: httpx.Response(200, json={"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}), monkeypatch)
    assert c.embed(["a", "b"]) == [[1.0], [2.0]] and c.embed([]) == []


@pytest.mark.parametrize("payload", [{"data": []}, {"data": [{"embedding": [1.0]}]}, {"data": [{"index": 0, "embedding": []}]}, {"nope": 1}])
def test_embed_rejects_malformed_payloads(monkeypatch, payload):
    c = client_with(lambda req: httpx.Response(200, json=payload), monkeypatch)
    with pytest.raises(OpenAIError):
        c.embed(["a"])


def test_embed_rejects_blank_input(monkeypatch):
    with pytest.raises(OpenAIError):
        client_with(lambda r: embedding_response([[1.0]]), monkeypatch).embed(["  "])


@pytest.mark.parametrize("status,needle", [(401, "authentication"), (403, "permission"), (404, "not found"), (429, "rate limit"), (500, "unavailable"), (400, "failed")])
def test_embed_maps_http_errors(monkeypatch, status, needle):
    c = client_with(lambda req: httpx.Response(status, json={"error": {"message": "detail here"}}), monkeypatch)
    with pytest.raises(OpenAIError) as exc:
        c.embed(["a"])
    assert needle in str(exc.value).lower() and "detail here" in str(exc.value)


def test_embed_maps_network_errors(monkeypatch):
    def raiser(exc):
        def handler(req):
            raise exc
        return handler
    for exc, needle in [(httpx.ConnectTimeout("t"), "timed out"), (httpx.ConnectError("c"), "connect"), (httpx.ReadError("r"), "failed")]:
        with pytest.raises(OpenAIError) as e:
            client_with(raiser(exc), monkeypatch).embed(["a"])
        assert needle in str(e.value).lower()


def test_embed_invalid_json_and_plain_text_errors(monkeypatch):
    with pytest.raises(OpenAIError, match="invalid JSON"):
        client_with(lambda r: httpx.Response(200, text="not json"), monkeypatch).embed(["a"])
    with pytest.raises(OpenAIError):
        client_with(lambda r: httpx.Response(500, text="server exploded"), monkeypatch).embed(["a"])


def test_is_available(monkeypatch):
    assert client_with(lambda r: httpx.Response(200, json={}), monkeypatch).is_available() is True
    assert client_with(lambda r: httpx.Response(401), monkeypatch).is_available() is False
    assert OpenAIClient("", "https://x", "c", "e").is_available() is False


# --------------------------------------------------------------------------------- glossary/loaders
def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


@pytest.mark.parametrize("name,text,match", [
    ("missing.json", None, "not found"), ("g.txt", "x", ".json"), ("g.json", "   ", "empty"), ("g.json", "{oops", "Invalid glossary JSON"),
    ("g.jsonl", '{"term":"a","definition":"b"}\nnot json', "line 2"), ("g.json", '[{"definition":"b"}]', "term"),
    ("g.json", '[{"term":"a"}]', "definition"), ("g.json", '[1, 2]', "No valid"), ("g.json", '{"x": 1}', "Expected a glossary array"),
])
def test_glossary_file_errors(tmp_path, name, text, match):
    path = tmp_path / name if text is None else write(tmp_path, name, text)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_glossary_file(path)


def test_glossary_wrappers_dedupe_jsonl_and_store_rules(tmp_path):
    wrapped = write(tmp_path, "w.json", json.dumps({"terms": [{"term": "ETF", "definition": " a  fund "}, {"term": "etf", "definition": "dup"}]}))
    entries = load_glossary_file(wrapped)
    assert [(e.term, e.definition) for e in entries] == [("ETF", "a fund")]
    single = write(tmp_path, "s.json", '{"term":"Bond","definition":"A loan."}')
    assert load_glossary_file(single)[0].term == "Bond"
    lines = write(tmp_path, "l.jsonl", '{"term":"A","definition":"x"}\n\n{"term":"B","definition":"y"}')
    store = GlossaryStore(tmp_path / "g.db")
    assert store.initialize_from_file(lines) == 2
    with pytest.raises(GlossaryAlreadyInitializedError):
        store.initialize(entries)
    with pytest.raises(ValueError):
        GlossaryStore(tmp_path / "g2.db").initialize([])
    assert [e.term for e in store.list("x")] == ["A"] and len(store.list(limit=0)) == 1


def test_loader_tolerates_trailing_commas_and_reports_bad_json():
    assert loaders._remove_trailing_commas('{"a": [1, 2,], "b": "x,]",}') == '{"a": [1, 2], "b": "x,]"}'
    assert loaders._parse_json('{"a": [1,],}', "f") == {"a": [1]}
    assert loaders._parse_json('{"a": "quote \\" , ]"}', "f") == {"a": 'quote " , ]'}
    with pytest.raises(ValueError, match="Invalid JSON in f"):
        loaders._parse_json("{oops", "f")
    with pytest.raises(ValueError, match="Invalid JSON in f"):
        loaders._parse_json('{"a": 1,, }', "f")


# ---------------------------------------------------------------------------------------------- CLIs
def test_seed_cli(monkeypatch, capsys, tmp_path):
    from src.data import seed as seed_mod
    monkeypatch.setattr("sys.argv", ["seed", "--reset"])
    monkeypatch.setattr(seed_mod, "settings", SimpleNamespace(app_db_path=tmp_path / "s.db"))
    assert seed_mod.main() == 0 and "Seeded 10" in capsys.readouterr().out


def test_ingest_cli_success_and_failure(monkeypatch, capsys, tmp_path):
    from src.rag import ingest as ing
    kb = SimpleNamespace(on_change=[], ingest=lambda chunks, reset=False: len(chunks), status=lambda: {"qdrant": "ok"})
    monkeypatch.setattr(ing, "build_knowledge_base", lambda embedder: kb)
    monkeypatch.setattr(ing, "settings", SimpleNamespace(openai_api_key="k", openai_base_url="https://x", openai_chat_model="c",
                                                         openai_embed_model="e", app_db_path=tmp_path / "i.db"))
    doc = tmp_path / "kb.json"
    doc.write_text(json.dumps([{"title": "Bonds", "content": "A bond is a loan. " * 8}]), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["ingest", "--file", str(doc), "--reset"])
    assert ing.main() == 0 and "Loaded 1 documents" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["ingest", "--file", str(tmp_path / "nope.json")])
    assert ing.main() == 1 and "Ingestion failed" in capsys.readouterr().err


def test_startup_warms_the_privacy_filter_and_survives_its_failure(monkeypatch, caplog):
    warmed = []
    monkeypatch.setattr(main, "redact_text", lambda text: warmed.append(text))
    with TestClient(main.app):
        pass
    assert warmed == ["warm up"]

    def boom(text):
        raise RuntimeError("model missing")
    monkeypatch.setattr(main, "redact_text", boom)
    with TestClient(main.app) as client:  # startup must not crash
        assert client.get("/api/stats").status_code == 200
