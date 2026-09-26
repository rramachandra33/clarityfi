from pathlib import Path

import pytest

from src.data.glossary_store import (
    GlossaryAlreadyInitializedError,
    GlossaryStore,
    load_glossary_file,
)


def test_loads_supplied_format_and_discards_extra_fields(tmp_path: Path):
    path = tmp_path / "financial_glossary.json"
    path.write_text(
        """
        {"term":"Asset","definition":"Something with economic value.","category":"Basics"},
        {"term":"Asset Allocation","definition":"Dividing investments among stocks, bonds and cash.","last_reviewed":"2026-09-15"}
        """,
        encoding="utf-8",
    )
    entries = load_glossary_file(path)
    assert [(entry.term, entry.definition) for entry in entries] == [
        ("Asset", "Something with economic value."),
        ("Asset Allocation", "Dividing investments among stocks, bonds and cash."),
    ]


def test_store_initializes_once_and_is_searchable(tmp_path: Path):
    fixture = Path(__file__).parents[2] / "data/financial_glossary.example.json"
    store = GlossaryStore(tmp_path / "glossary.db")
    assert store.initialize_from_file(fixture) == 2
    assert store.count() == 2
    assert [entry.term for entry in store.list("allocation")] == ["Asset Allocation"]

    with pytest.raises(GlossaryAlreadyInitializedError):
        store.initialize_from_file(fixture)
