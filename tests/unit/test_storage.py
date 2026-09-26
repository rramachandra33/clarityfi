from datetime import date

import pytest
from pydantic import ValidationError

from src.core.models import AssetClass, Goal, Holding, Portfolio
from src.data.db import Database
from src.data.repositories import (
    GoalRepository,
    MessageRepository,
    PasswordResetRepository,
    PortfolioRepository,
    SessionRepository,
    UserRepository,
)
from src.data.seed import SYNTHETIC_USERS, seed


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "t.db")
    database.init_schema()
    return database


def test_schema_init_is_idempotent(db):
    db.init_schema()
    with db.connect() as conn:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 7
    assert {"users", "sessions", "messages", "holdings", "portfolio_cash", "goals"} <= tables


def test_connect_rolls_back_on_error(db):
    with pytest.raises(RuntimeError):
        with db.connect() as conn:
            conn.execute("INSERT INTO users (user_id) VALUES ('abc-1')")
            raise RuntimeError("boom")
    assert UserRepository(db).get("abc-1") is None


def test_seed_creates_ten_distinct_synthetic_users(db):
    assert seed(db) == 10
    users = UserRepository(db)
    assert users.count() == 10
    assert len(users.list(synthetic_only=True)) == 10
    portfolios = PortfolioRepository(db)
    sizes = {spec["profile"].user_id: len(portfolios.get(spec["profile"].user_id).holdings) for spec in SYNTHETIC_USERS}
    assert sizes["learner-103"] == 0 and sizes["learner-110"] == 0  # edge cases: cash-only, empty
    assert sizes["learner-101"] >= 4
    assert len({s["profile"].user_id for s in SYNTHETIC_USERS}) == 10


def test_seed_is_idempotent_and_reset_only_touches_synthetic(db):
    seed(db)
    real, _ = UserRepository(db).ensure("my-real-id")
    PortfolioRepository(db).replace(Portfolio(user_id=real.user_id, holdings=[Holding(ticker="vti", shares=1)]))
    PortfolioRepository(db).replace(Portfolio(user_id="learner-101", holdings=[]))  # user edits a demo portfolio
    assert seed(db) == 0  # nothing re-written, edit preserved
    assert PortfolioRepository(db).get("learner-101").holdings == []
    assert seed(db, reset=True) == 10  # reset rebuilds demo data
    assert len(PortfolioRepository(db).get("learner-101").holdings) >= 4
    assert PortfolioRepository(db).get("my-real-id").holdings[0].ticker == "VTI"  # untouched
    assert UserRepository(db).is_synthetic("my-real-id") is False


def test_seed_goals_saved(db):
    seed(db)
    goals = GoalRepository(db)
    assert len(goals.list("learner-108")) == 2
    assert goals.list("learner-110") == []


def test_set_email_and_find_by_email_are_case_insensitive_and_unique(db):
    users = UserRepository(db)
    users.ensure("alice"), users.ensure("bob")
    assert users.set_email("alice", "Alice@Example.com") is True
    assert users.find_id_by_email("alice@example.com") == "alice"  # case-insensitive lookup
    assert users.find_id_by_email("ALICE@EXAMPLE.COM") == "alice"
    assert users.set_email("bob", "alice@example.com") is False  # already taken (case-insensitively)
    assert users.set_email("ghost", "x@example.com") is False  # unknown user id
    assert users.set_email("alice", None) is True  # clearing it frees the address up
    assert users.set_email("bob", "alice@example.com") is True
    assert users.find_id_by_email("nobody@example.com") is None


def test_password_reset_tokens_are_single_use_time_limited_and_one_live_per_user(db):
    users = UserRepository(db)
    users.ensure("carol")
    resets = PasswordResetRepository(db)
    assert resets.peek("unknown-hash") is None

    resets.create("carol", "hash-1", ttl_seconds=3600)
    assert resets.peek("hash-1") == "carol"
    assert resets.consume("hash-1") == "carol"
    assert resets.peek("hash-1") is None  # used
    assert resets.consume("hash-1") is None  # consuming twice fails

    resets.create("carol", "hash-2", ttl_seconds=-1)  # already expired
    assert resets.peek("hash-2") is None
    assert resets.consume("hash-2") is None

    resets.create("carol", "hash-3", ttl_seconds=3600)
    resets.create("carol", "hash-4", ttl_seconds=3600)  # a second request supersedes the first
    assert resets.peek("hash-3") is None
    assert resets.peek("hash-4") == "carol"


def test_goal_find_by_name_and_update_persist_assumed_return(db):
    UserRepository(db).ensure("learner-777")
    goals = GoalRepository(db)
    assert goals.find_id_by_name("learner-777", "Car") is None

    gid = goals.add("learner-777", Goal(name="Car", target_amount=10000, target_date=date(2030, 1, 1), assumed_return_pct=2))
    assert goals.find_id_by_name("learner-777", "car") == gid  # case/whitespace-insensitive
    assert goals.find_id_by_name("learner-777", " Car ") == gid

    ok = goals.update("learner-777", gid, Goal(name="Car", target_amount=18000, target_date=date(2031, 1, 1), assumed_return_pct=6))
    assert ok is True
    [(row_id, goal)] = goals.list_rows("learner-777")
    assert row_id == gid and goal.target_amount == 18000 and goal.assumed_return_pct == 6

    assert goals.update("learner-777", 999999, Goal(name="Nope", target_amount=1, target_date=date(2030, 1, 1))) is False


def test_ensure_saves_new_id_once_and_is_case_insensitive(db):
    users = UserRepository(db)
    profile, created = users.ensure("Learner-204")
    assert created and profile.user_id == "Learner-204" and profile.risk_tolerance.value == "moderate"
    again, created_again = users.ensure("learner-204")
    assert not created_again and again.user_id == "Learner-204"  # canonical id preserved
    assert users.count() == 1


def test_ensure_returns_existing_synthetic_user_unchanged(db):
    seed(db)
    profile, created = UserRepository(db).ensure("LEARNER-101")
    assert not created and profile.name.startswith("Alex")
    assert UserRepository(db).count() == 10


@pytest.mark.parametrize("bad", ["ab", "", "   ", "x" * 121, "bad;id", "<script>alert(1)</script>", "a' OR '1'='1"])
def test_ensure_rejects_invalid_ids(db, bad):
    with pytest.raises(ValidationError):
        UserRepository(db).ensure(bad)
    assert UserRepository(db).count() == 0


def test_update_profile(db):
    users = UserRepository(db)
    users.ensure("learner-999")
    updated = users.update_profile("learner-999", risk_tolerance="aggressive", horizon_years=40)
    assert updated.risk_tolerance.value == "aggressive" and updated.horizon_years == 40
    with pytest.raises(ValidationError):
        users.update_profile("learner-999", horizon_years=0)
    assert users.update_profile("nobody-here") is None


def test_sessions_and_message_history(db):
    UserRepository(db).ensure("learner-500")
    sessions, messages = SessionRepository(db), MessageRepository(db)
    sid = sessions.create("learner-500")
    assert sessions.get_user(sid) == "learner-500" and sessions.list_for_user("learner-500") == [sid]
    for i in range(5):
        messages.add(sid, "user", f"q{i}")
    messages.add(sid, "assistant", "a", agent="finance_qa", citations=[{"index": 1, "title": "ETF"}], cache_hit="exact")
    history = messages.history(sid, limit=3)
    assert [m["content"] for m in history] == ["q3", "q4", "a"]  # newest 3, oldest first
    assert history[-1]["citations"][0]["title"] == "ETF" and history[-1]["cache_hit"] == "exact"
    assert messages.history("missing") == []


def test_message_role_constraint_and_foreign_key(db):
    UserRepository(db).ensure("learner-501")
    sid = SessionRepository(db).create("learner-501")
    with pytest.raises(Exception):
        MessageRepository(db).add(sid, "hacker", "x")
    with pytest.raises(Exception):
        MessageRepository(db).add("no-such-session", "user", "x")
    with pytest.raises(Exception):
        SessionRepository(db).create("ghost-user")


def test_deleting_user_cascades(db):
    seed(db)
    with db.connect() as conn:
        conn.execute("DELETE FROM users WHERE user_id = 'learner-108'")
    assert PortfolioRepository(db).get("learner-108").holdings == []
    assert GoalRepository(db).list("learner-108") == []


def test_portfolio_replace_is_atomic_and_normalizes(db):
    UserRepository(db).ensure("learner-777")
    repo = PortfolioRepository(db)
    repo.replace(Portfolio(user_id="learner-777", cash=50, holdings=[Holding(ticker="aapl", shares=2, asset_class=AssetClass.stock)]))
    repo.replace(Portfolio(user_id="learner-777", cash=75, holdings=[Holding(ticker="bnd", shares=5, asset_class=AssetClass.bond, expense_ratio=0.0003)]))
    result = repo.get("learner-777")
    assert [h.ticker for h in result.holdings] == ["BND"] and result.cash == 75


def test_negative_values_rejected_by_db(db):
    UserRepository(db).ensure("learner-778")
    with pytest.raises(Exception):
        with db.connect() as conn:
            conn.execute("INSERT INTO holdings (user_id, ticker, shares) VALUES ('learner-778','X',-1)")


def test_in_memory_database_smoke():
    mem = Database(":memory:")
    mem.init_schema()  # each connect() is a fresh :memory: db; only verify it does not error


def test_v4_database_gains_password_and_session_tables(tmp_path):
    import sqlite3

    from src.data.db import Database

    path = tmp_path / "v4.db"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE users (user_id TEXT PRIMARY KEY COLLATE NOCASE, name TEXT, risk_tolerance TEXT NOT NULL DEFAULT 'moderate',
                experience_level TEXT NOT NULL DEFAULT 'beginner', horizon_years INTEGER NOT NULL DEFAULT 10,
                is_synthetic INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT 'x');
            INSERT INTO users (user_id) VALUES ('kept-user');
            PRAGMA user_version = 4;""")
    db = Database(path)
    db.init_schema()
    with db.connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
        assert "password_hash" in cols and conn.execute("SELECT user_id FROM users").fetchone()[0] == "kept-user"
        assert conn.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 7


def test_v5_database_gains_assumed_return_pct_on_goals(tmp_path):
    import sqlite3

    from src.data.db import Database

    path = tmp_path / "v5.db"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE users (user_id TEXT PRIMARY KEY COLLATE NOCASE, name TEXT, risk_tolerance TEXT NOT NULL DEFAULT 'moderate',
                experience_level TEXT NOT NULL DEFAULT 'beginner', horizon_years INTEGER NOT NULL DEFAULT 10,
                is_synthetic INTEGER NOT NULL DEFAULT 0, password_hash TEXT, created_at TEXT NOT NULL DEFAULT 'x');
            CREATE TABLE goals (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, name TEXT NOT NULL,
                target_amount REAL NOT NULL, target_date TEXT NOT NULL, current_savings REAL NOT NULL DEFAULT 0,
                monthly_contribution REAL NOT NULL DEFAULT 0, risk_tolerance TEXT NOT NULL DEFAULT 'moderate');
            INSERT INTO users (user_id) VALUES ('kept-user');
            INSERT INTO goals (user_id, name, target_amount, target_date) VALUES ('kept-user', 'Old goal', 5000, '2030-01-01');
            PRAGMA user_version = 5;""")
    db = Database(path)
    db.init_schema()
    with db.connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(goals)")}
        assert "assumed_return_pct" in cols
        row = conn.execute("SELECT * FROM goals WHERE user_id = 'kept-user'").fetchone()
        assert row["name"] == "Old goal" and row["assumed_return_pct"] == 0  # pre-existing row keeps its data, gets the new column's default
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 7
