import sqlite3
import threading

import pytest

from src.data.db import Database
from src.data.repositories import MessageRepository, SessionRepository, UserRepository
from src.workflow.memory import SqliteConversationMemory


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "mem.db")
    d.init_schema()
    return d


@pytest.fixture
def mem(db):
    return SqliteConversationMemory(db)


def test_unknown_user_or_conversation_has_empty_history_and_creates_nothing(mem, db):
    assert mem.history(mem.key("nobody-here", "c1")) == []
    UserRepository(db).ensure("learner-204")
    assert mem.history(mem.key("learner-204", "never-used")) == []
    assert UserRepository(db).count() == 1  # reading never creates users or sessions


def test_exchange_round_trip_preserves_order_and_roles(mem):
    key = mem.key("learner-204", "c1")
    assert mem.record_exchange(key, "What is an ETF?", "A basket of investments.") is True
    mem.record_exchange(key, "And bonds?", "Loans to issuers.")
    assert mem.history(key) == [
        {"role": "user", "content": "What is an ETF?"}, {"role": "assistant", "content": "A basket of investments."},
        {"role": "user", "content": "And bonds?"}, {"role": "assistant", "content": "Loans to issuers."}]


def test_history_is_capped_to_the_most_recent_messages(db):
    mem = SqliteConversationMemory(db, max_messages=4)
    key = mem.key("learner-204", "c1")
    for i in range(5):
        mem.record_exchange(key, f"q{i}", f"a{i}")
    assert [t["content"] for t in mem.history(key)] == ["q3", "a3", "q4", "a4"]


def test_history_survives_a_restart(db):
    """The gap flagged earlier: chat history used to vanish when the server restarted."""
    first = SqliteConversationMemory(db)
    key = first.key("learner-204", "finance-qa")
    first.record_exchange(key, "What is an ETF?", "A basket.")
    restarted = SqliteConversationMemory(Database(db.path))  # a brand-new process would build these afresh
    assert restarted.history(key)[0]["content"] == "What is an ETF?"


def test_users_and_conversations_are_isolated(mem):
    mem.record_exchange(mem.key("learner-204", "c1"), "mine", "answer 1")
    mem.record_exchange(mem.key("learner-205", "c1"), "theirs", "answer 2")
    mem.record_exchange(mem.key("learner-204", "c2"), "other chat", "answer 3")
    assert [t["content"] for t in mem.history(mem.key("learner-204", "c1"))] == ["mine", "answer 1"]
    assert [t["content"] for t in mem.history(mem.key("learner-205", "c1"))] == ["theirs", "answer 2"]
    assert [t["content"] for t in mem.history(mem.key("learner-204", "c2"))] == ["other chat", "answer 3"]


def test_user_id_case_is_normalized_to_the_canonical_id(mem, db):
    mem.record_exchange(mem.key("Learner-204", "c1"), "q", "a")
    assert mem.history(mem.key("learner-204", "c1"))[0]["content"] == "q"
    assert UserRepository(db).count() == 1


def test_unstorable_user_ids_and_blank_content_are_skipped(mem):
    assert mem.record_exchange(mem.key("bad;id<script>", "c1"), "q", "a") is False
    assert mem.record_exchange(mem.key("learner-204", "c1"), "", "a") is False
    assert mem.record_exchange(mem.key("learner-204", "c1"), "q", "   ") is False
    assert mem.history(mem.key("learner-204", "c1")) == []


def test_agent_citations_and_cache_metadata_are_stored_alongside_the_message(mem, db):
    key = mem.key("learner-204", "c1")
    mem.record_exchange(key, "What is an ETF?", "A basket.", agent="finance_qa", cache_hit="exact",
                        citations=[{"index": 1, "title": "ETF", "url": "https://ex.com/etf"}])
    sid = SessionRepository(db).find("learner-204", "c1")
    stored = MessageRepository(db).history(sid)
    assert stored[1]["agent"] == "finance_qa" and stored[1]["cache_hit"] == "exact" and stored[1]["citations"][0]["title"] == "ETF"
    assert stored[0]["agent"] is None


def test_conversation_ids_containing_colons_are_handled(mem):
    key = mem.key("learner-204", "tab:finance:1")
    mem.record_exchange(key, "q", "a")
    assert mem.history(key)[0]["content"] == "q"


def test_clear_one_conversation_or_everything(mem):
    a, b = mem.key("learner-204", "c1"), mem.key("learner-204", "c2")
    mem.record_exchange(a, "q1", "a1")
    mem.record_exchange(b, "q2", "a2")
    mem.clear(a)
    assert mem.history(a) == [] and mem.history(b)
    mem.clear("unknown-user:c1")  # clearing something that does not exist is a no-op
    mem.clear()
    assert mem.history(b) == []


# ------------------------------------------------------------------------------ sessions & schema
def test_get_or_create_is_idempotent_and_distinguishes_conversations(db):
    UserRepository(db).ensure("learner-204")
    s = SessionRepository(db)
    a1, a2, b = s.get_or_create("learner-204", "c1"), s.get_or_create("learner-204", "c1"), s.get_or_create("learner-204", "c2")
    assert a1 == a2 and a1 != b and s.find("learner-204", "c1") == a1 and s.find("learner-204", "zzz") is None


def test_concurrent_get_or_create_yields_a_single_session(db):
    UserRepository(db).ensure("learner-204")
    s = SessionRepository(db)
    found = []
    threads = [threading.Thread(target=lambda: found.append(s.get_or_create("learner-204", "race"))) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(set(found)) == 1
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE conversation_key='race'").fetchone()[0] == 1


def test_a_second_session_for_the_same_conversation_is_rejected_by_the_database(db):
    UserRepository(db).ensure("learner-204")
    SessionRepository(db).get_or_create("learner-204", "c1")
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect() as conn:
            conn.execute("INSERT INTO sessions (session_id, user_id, conversation_key) VALUES ('dup', 'learner-204', 'c1')")


def test_legacy_sessions_without_a_conversation_key_are_allowed_many_times(db):
    s = SessionRepository(db)
    UserRepository(db).ensure("learner-204")
    assert s.create("learner-204") != s.create("learner-204")  # NULL keys are not unique-constrained


def test_v3_database_is_migrated_in_place_without_losing_data(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE users (user_id TEXT PRIMARY KEY COLLATE NOCASE, name TEXT, risk_tolerance TEXT NOT NULL DEFAULT 'moderate',
            experience_level TEXT NOT NULL DEFAULT 'beginner', horizon_years INTEGER NOT NULL DEFAULT 10,
            is_synthetic INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT 'x');
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT 'x',
            last_active_at TEXT NOT NULL DEFAULT 'x');
        INSERT INTO users(user_id) VALUES ('legacy-user'); INSERT INTO sessions(session_id, user_id) VALUES ('old-session', 'legacy-user');
        PRAGMA user_version = 3;""")
    con.commit()
    con.close()
    db = Database(path)
    db.init_schema()
    db.init_schema()  # running the migration twice is harmless
    with db.connect() as conn:
        assert "conversation_key" in {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
        assert conn.execute("SELECT session_id FROM sessions").fetchone()[0] == "old-session"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 7
    assert SqliteConversationMemory(db).record_exchange(SqliteConversationMemory.key("legacy-user", "c1"), "q", "a")


def test_persistence_end_to_end_through_the_workflow(db):
    """A follow-up after a simulated restart still sees the earlier turn."""
    import asyncio

    from src.workflow.graph import WorkflowRequest
    from tests.unit._agent_fakes import FakeAgent, make_workflow

    UserRepository(db).ensure("learner-204")
    agent = FakeAgent("finance_qa", "An ETF is a basket.")
    asyncio.run(make_workflow([agent], memory=SqliteConversationMemory(db)).ainvoke(
        WorkflowRequest("What is an ETF?", "learner-204", "finance-qa")))
    agent2 = FakeAgent("finance_qa", "Bonds are loans.")
    asyncio.run(make_workflow([agent2], memory=SqliteConversationMemory(Database(db.path))).ainvoke(
        WorkflowRequest("And what about bonds?", "learner-204", "finance-qa")))
    assert [t["content"] for t in agent2.requests[0].history][0] == "What is an ETF?"
