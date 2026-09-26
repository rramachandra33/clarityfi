"""SQLite access for users, sessions, messages, portfolios and goals.

One short-lived connection per operation keeps this safe under FastAPI's threadpool. WAL mode
allows concurrent readers while a write is in flight. The schema is created idempotently.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 7

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id           TEXT PRIMARY KEY COLLATE NOCASE,
    name              TEXT,
    risk_tolerance    TEXT NOT NULL DEFAULT 'moderate',
    experience_level  TEXT NOT NULL DEFAULT 'beginner',
    horizon_years     INTEGER NOT NULL DEFAULT 10,
    is_synthetic      INTEGER NOT NULL DEFAULT 0,
    password_hash     TEXT,   -- scrypt$n$r$p$salt$hash (NULL = no password set yet)
    email             TEXT COLLATE NOCASE,   -- optional; only self-registered users can set one, for password reset
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
-- The unique index on email is created in _migrate(), not here: a pre-existing database reaches this
-- CREATE TABLE IF NOT EXISTS as a no-op (table already exists), so the column may not exist yet at this point.

CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash  TEXT PRIMARY KEY,   -- sha256 of the cookie token; the token itself is never stored
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);

-- Password-reset tokens. Only the SHA-256 of the raw token is ever stored (same pattern as auth_sessions);
-- the raw token exists only in the email sent to the user, and briefly in memory while handling the request.
CREATE TABLE IF NOT EXISTS password_resets (
    token_hash  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    used_at     TEXT   -- NULL until consumed; a used or expired token is rejected
);
CREATE INDEX IF NOT EXISTS idx_password_resets_user ON password_resets(user_id);

CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    conversation_key TEXT,   -- the client's conversation id; one session per (user, conversation)
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    last_active_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- content is stored REDACTED (post-Presidio); never write raw user text here.
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
    content         TEXT NOT NULL,
    agent           TEXT,
    citations_json  TEXT,
    cache_hit       TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS holdings (
    user_id        TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    ticker         TEXT NOT NULL,
    shares         REAL NOT NULL CHECK (shares >= 0),
    cost_basis     REAL,
    asset_class    TEXT NOT NULL DEFAULT 'stock',
    expense_ratio  REAL,
    PRIMARY KEY (user_id, ticker)
);

CREATE TABLE IF NOT EXISTS portfolio_cash (
    user_id  TEXT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    cash     REAL NOT NULL DEFAULT 0 CHECK (cash >= 0)
);

CREATE TABLE IF NOT EXISTS goals (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id               TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name                  TEXT NOT NULL,
    target_amount         REAL NOT NULL CHECK (target_amount > 0),
    target_date           TEXT NOT NULL,
    current_savings       REAL NOT NULL DEFAULT 0,
    monthly_contribution  REAL NOT NULL DEFAULT 0,
    risk_tolerance        TEXT NOT NULL DEFAULT 'moderate',
    assumed_return_pct    REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id);

-- Persistent API-response cache (survives restarts, so free-tier quotas are not burned on reboot).
CREATE TABLE IF NOT EXISTS kv_cache (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    stored_at  REAL NOT NULL
);

-- Finance Q&A answer cache (exact + semantic). Questions are stored REDACTED (post-Presidio) and only
-- self-contained questions are cached; entries are tied to a knowledge-base version.
CREATE TABLE IF NOT EXISTS qa_cache (
    key_hash      TEXT PRIMARY KEY,
    scope         TEXT NOT NULL DEFAULT '',
    kb_version    TEXT NOT NULL,
    question_norm TEXT NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    sources_json  TEXT NOT NULL,
    embedding     BLOB,
    created_at    REAL NOT NULL,
    last_used_at  REAL NOT NULL,
    hit_count     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_qa_scope ON qa_cache(scope, kb_version);

-- Per-provider request counts per period ('YYYY-MM-DD' for daily, 'YYYY-MM' for monthly quotas).
CREATE TABLE IF NOT EXISTS api_usage (
    provider  TEXT NOT NULL,
    period    TEXT NOT NULL,
    count     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, period)
);
"""


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection that commits on success and rolls back on error."""
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            self._migrate(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Upgrade databases created by older schema versions (CREATE IF NOT EXISTS cannot add columns)."""
        user_columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        if "password_hash" not in user_columns:  # v4 -> v5
            conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
        goal_columns = {row["name"] for row in conn.execute("PRAGMA table_info(goals)")}
        if "assumed_return_pct" not in goal_columns:  # v5 -> v6
            conn.execute("ALTER TABLE goals ADD COLUMN assumed_return_pct REAL NOT NULL DEFAULT 0")
        if "email" not in user_columns:  # v6 -> v7
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT COLLATE NOCASE")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL")
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "conversation_key" not in columns:  # v3 -> v4
            conn.execute("ALTER TABLE sessions ADD COLUMN conversation_key TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_conversation "
                     "ON sessions(user_id, conversation_key) WHERE conversation_key IS NOT NULL")
