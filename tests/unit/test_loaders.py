import json

from src.rag.loaders import chunk_documents, load_json_documents


def test_loads_nested_category_arrays(tmp_path):
    path = tmp_path / "knowledge.json"
    path.write_text(
        json.dumps(
            {
                "stocks": [
                    {"title": "Index funds", "body": "An index fund follows a defined market index and holds a representative basket of securities."}
                ],
                "glossary": {"duration": "Duration estimates how sensitive a bond price may be to changes in interest rates."},
            }
        ),
        encoding="utf-8",
    )

    documents = load_json_documents(path)
    assert len(documents) == 2
    assert documents[0].metadata["category"] == "Stocks"
    assert {document.title for document in documents} == {"Index funds", "Duration"}


def test_loads_jsonl_question_answers(tmp_path):
    path = tmp_path / "faq.jsonl"
    rows = [
        {"question": "What is an ETF?", "answer": "An ETF is a pooled investment that trades on an exchange."},
        {"question": "What is a bond?", "answer": "A bond is a debt security issued by a government or company."},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    documents = load_json_documents(path)
    assert len(documents) == 2
    assert documents[0].text.startswith("Question:")


def test_chunks_long_documents(tmp_path):
    path = tmp_path / "long.json"
    path.write_text(json.dumps([{"title": "Long", "content": "finance " * 500}]), encoding="utf-8")
    chunks = chunk_documents(load_json_documents(path), chunk_size=300, overlap=40)
    assert len(chunks) > 5
    assert all(len(chunk.text) <= 300 for chunk in chunks)
