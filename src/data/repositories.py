"""Repositories over the SQLite schema. They return the shared pydantic models."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.core.models import (
    ExperienceLevel,
    Goal,
    Holding,
    Portfolio,
    RiskTolerance,
    UserProfile,
)
from src.data.db import Database


def _profile(row: sqlite3.Row) -> UserProfile:
    return UserProfile(
        user_id=row["user_id"],
        name=row["name"],
        risk_tolerance=row["risk_tolerance"],
        experience_level=row["experience_level"],
        horizon_years=row["horizon_years"],
        email=row["email"] if "email" in row.keys() else None,
    )


class UserRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, user_id: str) -> UserProfile | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id.strip(),)).fetchone()
        return _profile(row) if row else None

    def is_synthetic(self, user_id: str) -> bool:
        with self.db.connect() as conn:
            row = conn.execute("SELECT is_synthetic FROM users WHERE user_id = ?", (user_id.strip(),)).fetchone()
        return bool(row and row["is_synthetic"])

    def list(self, *, synthetic_only: bool = False) -> list[UserProfile]:
        sql = "SELECT * FROM users" + (" WHERE is_synthetic = 1" if synthetic_only else "") + " ORDER BY user_id"
        with self.db.connect() as conn:
            return [_profile(r) for r in conn.execute(sql)]

    def count(self) -> int:
        with self.db.connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def ensure(self, user_id: str) -> tuple[UserProfile, bool]:
        """Return (profile, created). Known ids (any letter case) are returned unchanged;
        an unknown id is saved as a new user with a default profile and empty portfolio."""
        candidate = UserProfile(user_id=user_id.strip())  # validates length/charset
        with self.db.connect() as conn:
            cur = conn.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (candidate.user_id,))
            created = cur.rowcount == 1
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (candidate.user_id,)).fetchone()
        return _profile(row), created

    def upsert_synthetic(self, profile: UserProfile) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """INSERT INTO users (user_id, name, risk_tolerance, experience_level, horizon_years, is_synthetic)
                   VALUES (?, ?, ?, ?, ?, 1)
                   ON CONFLICT(user_id) DO UPDATE SET name=excluded.name, risk_tolerance=excluded.risk_tolerance,
                       experience_level=excluded.experience_level, horizon_years=excluded.horizon_years,
                       is_synthetic=1""",
                (profile.user_id, profile.name, profile.risk_tolerance.value,
                 profile.experience_level.value, profile.horizon_years),
            )

    def update_profile(
        self,
        user_id: str,
        *,
        risk_tolerance: RiskTolerance | str | None = None,
        experience_level: ExperienceLevel | str | None = None,
        horizon_years: int | None = None,
    ) -> UserProfile | None:
        fields: dict[str, Any] = {}
        if risk_tolerance:
            fields["risk_tolerance"] = RiskTolerance(risk_tolerance).value
        if experience_level:
            fields["experience_level"] = ExperienceLevel(experience_level).value
        if horizon_years is not None:
            fields["horizon_years"] = UserProfile(user_id=user_id, horizon_years=horizon_years).horizon_years
        if fields:
            assignments = ", ".join(f"{k} = ?" for k in fields)
            with self.db.connect() as conn:
                conn.execute(f"UPDATE users SET {assignments} WHERE user_id = ?", (*fields.values(), user_id.strip()))
        return self.get(user_id)

    def set_email(self, user_id: str, email: str | None) -> bool:
        """False if another account already has this email (case-insensitive) or the user id is unknown."""
        with self.db.connect() as conn:
            try:
                cur = conn.execute("UPDATE users SET email = ? WHERE user_id = ?", (email, user_id.strip()))
            except sqlite3.IntegrityError:
                return False
            return cur.rowcount == 1

    def find_id_by_email(self, email: str) -> str | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT user_id FROM users WHERE email = ?", (email.strip(),)).fetchone()
        return row["user_id"] if row else None


class SessionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, user_id: str) -> str:
        session_id = uuid.uuid4().hex
        with self.db.connect() as conn:
            conn.execute("INSERT INTO sessions (session_id, user_id) VALUES (?, ?)", (session_id, user_id.strip()))
        return session_id

    def get_or_create(self, user_id: str, conversation_key: str) -> str:
        """The session for (user, conversation), created on first use. Safe under concurrent callers."""
        uid, key = user_id.strip(), conversation_key.strip()
        for _ in range(2):
            with self.db.connect() as conn:
                row = conn.execute("SELECT session_id FROM sessions WHERE user_id = ? AND conversation_key = ?", (uid, key)).fetchone()
                if row:
                    return row["session_id"]
                try:
                    session_id = uuid.uuid4().hex
                    conn.execute("INSERT INTO sessions (session_id, user_id, conversation_key) VALUES (?, ?, ?)", (session_id, uid, key))
                    return session_id
                except sqlite3.IntegrityError:
                    continue  # another caller created it first: read it on the next pass
        raise RuntimeError("could not create or find the conversation session")

    def find(self, user_id: str, conversation_key: str) -> str | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT session_id FROM sessions WHERE user_id = ? AND conversation_key = ?",
                               (user_id.strip(), conversation_key.strip())).fetchone()
        return row["session_id"] if row else None

    def get_user(self, session_id: str) -> str | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row["user_id"] if row else None

    def list_for_user(self, user_id: str) -> list[str]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions WHERE user_id = ? ORDER BY last_active_at DESC", (user_id.strip(),)
            )
            return [r["session_id"] for r in rows]

    def touch(self, session_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE sessions SET last_active_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE session_id = ?",
                (session_id,),
            )


class MessageRepository:
    """Chat history. `content` MUST already be redacted by the PII layer."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def add(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        agent: str | None = None,
        citations: list[dict[str, Any]] | None = None,
        cache_hit: str | None = None,
    ) -> int:
        with self.db.connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, agent, citations_json, cache_hit) VALUES (?,?,?,?,?,?)",
                (session_id, role, content, agent, json.dumps(citations) if citations else None, cache_hit),
            )
            conn.execute(
                "UPDATE sessions SET last_active_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE session_id = ?",
                (session_id,),
            )
            return int(cur.lastrowid)

    def history(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent `limit` messages, oldest first."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id",
                (session_id, limit),
            ).fetchall()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "agent": r["agent"],
                "citations": json.loads(r["citations_json"]) if r["citations_json"] else [],
                "cache_hit": r["cache_hit"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]


class PortfolioRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, user_id: str) -> Portfolio:
        uid = user_id.strip()
        with self.db.connect() as conn:
            holdings = [
                Holding(
                    ticker=r["ticker"], shares=r["shares"], cost_basis=r["cost_basis"],
                    asset_class=r["asset_class"], expense_ratio=r["expense_ratio"],
                )
                for r in conn.execute("SELECT * FROM holdings WHERE user_id = ? ORDER BY ticker", (uid,))
            ]
            cash_row = conn.execute("SELECT cash FROM portfolio_cash WHERE user_id = ?", (uid,)).fetchone()
        return Portfolio(user_id=uid, holdings=holdings, cash=cash_row["cash"] if cash_row else 0.0)

    def replace(self, portfolio: Portfolio) -> None:
        """Atomically replace a user's holdings and cash."""
        uid = portfolio.user_id.strip()
        with self.db.connect() as conn:
            conn.execute("DELETE FROM holdings WHERE user_id = ?", (uid,))
            conn.executemany(
                "INSERT INTO holdings (user_id, ticker, shares, cost_basis, asset_class, expense_ratio) VALUES (?,?,?,?,?,?)",
                [(uid, h.ticker, h.shares, h.cost_basis, h.asset_class.value, h.expense_ratio) for h in portfolio.holdings],
            )
            conn.execute(
                "INSERT INTO portfolio_cash (user_id, cash) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET cash = excluded.cash",
                (uid, portfolio.cash),
            )


class GoalRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, user_id: str, goal: Goal) -> int:
        with self.db.connect() as conn:
            cur = conn.execute(
                """INSERT INTO goals (user_id, name, target_amount, target_date, current_savings,
                   monthly_contribution, risk_tolerance, assumed_return_pct) VALUES (?,?,?,?,?,?,?,?)""",
                (user_id.strip(), goal.name, goal.target_amount, goal.target_date.isoformat(),
                 goal.current_savings, goal.monthly_contribution, goal.risk_tolerance.value, goal.assumed_return_pct),
            )
            return int(cur.lastrowid)

    def find_id_by_name(self, user_id: str, name: str) -> int | None:
        """The id of this user's existing goal with the same name (case/whitespace-insensitive), if any."""
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT id FROM goals WHERE user_id = ? AND TRIM(LOWER(name)) = TRIM(LOWER(?)) ORDER BY id LIMIT 1",
                (user_id.strip(), name),
            ).fetchone()
        return int(row["id"]) if row else None

    def update(self, user_id: str, goal_id: int, goal: Goal) -> bool:
        with self.db.connect() as conn:
            return conn.execute(
                """UPDATE goals SET name = ?, target_amount = ?, target_date = ?, current_savings = ?,
                   monthly_contribution = ?, risk_tolerance = ?, assumed_return_pct = ? WHERE id = ? AND user_id = ?""",
                (goal.name, goal.target_amount, goal.target_date.isoformat(), goal.current_savings,
                 goal.monthly_contribution, goal.risk_tolerance.value, goal.assumed_return_pct, goal_id, user_id.strip()),
            ).rowcount > 0

    def list(self, user_id: str) -> list[Goal]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM goals WHERE user_id = ? ORDER BY id", (user_id.strip(),)).fetchall()
        return [
            Goal(
                name=r["name"], target_amount=r["target_amount"], target_date=date.fromisoformat(r["target_date"]),
                current_savings=r["current_savings"], monthly_contribution=r["monthly_contribution"],
                risk_tolerance=r["risk_tolerance"], assumed_return_pct=r["assumed_return_pct"],
            )
            for r in rows
        ]

    def list_rows(self, user_id: str) -> list[tuple[int, Goal]]:
        """[(goal id, Goal)] so a client can delete a specific goal."""
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM goals WHERE user_id = ? ORDER BY id", (user_id.strip(),)).fetchall()
        return [(r["id"], Goal(name=r["name"], target_amount=r["target_amount"], target_date=date.fromisoformat(r["target_date"]),
                               current_savings=r["current_savings"], monthly_contribution=r["monthly_contribution"],
                               risk_tolerance=r["risk_tolerance"], assumed_return_pct=r["assumed_return_pct"])) for r in rows]

    def delete(self, user_id: str, goal_id: int) -> bool:
        with self.db.connect() as conn:
            return conn.execute("DELETE FROM goals WHERE id = ? AND user_id = ?", (goal_id, user_id.strip())).rowcount > 0

    def count(self, user_id: str) -> int:
        with self.db.connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM goals WHERE user_id = ?", (user_id.strip(),)).fetchone()[0]

    def delete_all(self, user_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM goals WHERE user_id = ?", (user_id.strip(),))


class AuthRepository:
    """Password hashes and login sessions. Tokens are stored only as SHA-256 hashes."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def get_hash(self, user_id: str) -> str | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT password_hash FROM users WHERE user_id = ?", (user_id.strip(),)).fetchone()
        return row["password_hash"] if row else None

    def set_password_if_unset(self, user_id: str, password_hash: str) -> bool:
        """Atomically claim an account that has no password yet. False if it already has one (or does not exist)."""
        with self.db.connect() as conn:
            cur = conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ? AND password_hash IS NULL", (password_hash, user_id.strip()))
            return cur.rowcount == 1

    def set_password(self, user_id: str, password_hash: str) -> None:
        """Unconditionally overwrite the password (a real reset, not the 'claim an unset account' path above)."""
        with self.db.connect() as conn:
            conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (password_hash, user_id.strip()))

    def users_without_password(self, *, synthetic_only: bool = False) -> list[str]:
        sql = "SELECT user_id FROM users WHERE password_hash IS NULL" + (" AND is_synthetic = 1" if synthetic_only else "")
        with self.db.connect() as conn:
            return [r["user_id"] for r in conn.execute(sql)]

    def create_session(self, user_id: str, token_hash: str, ttl_seconds: int) -> None:
        now = datetime.now(timezone.utc)
        with self.db.connect() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (now.isoformat(),))  # tidy expired rows
            conn.execute("INSERT INTO auth_sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                         (token_hash, user_id, now.isoformat(), (now + timedelta(seconds=ttl_seconds)).isoformat()))

    def user_for_session(self, token_hash: str) -> str | None:
        """The canonical user id for a live session, else None (unknown or expired)."""
        with self.db.connect() as conn:
            row = conn.execute("SELECT user_id, expires_at FROM auth_sessions WHERE token_hash = ?", (token_hash,)).fetchone()
            if row is None:
                return None
            if row["expires_at"] < datetime.now(timezone.utc).isoformat():
                conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))
                return None
            user = conn.execute("SELECT user_id FROM users WHERE user_id = ?", (row["user_id"],)).fetchone()
        return user["user_id"] if user else None

    def delete_session(self, token_hash: str) -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))

    def delete_user_sessions(self, user_id: str) -> int:
        with self.db.connect() as conn:
            return conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,)).rowcount


class PasswordResetRepository:
    """Reset tokens, stored only as a SHA-256 hash — same pattern as auth_sessions. Single-use, time-limited."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, user_id: str, token_hash: str, ttl_seconds: int) -> None:
        now = datetime.now(timezone.utc)
        with self.db.connect() as conn:
            conn.execute("DELETE FROM password_resets WHERE expires_at < ?", (now.isoformat(),))  # tidy expired rows
            conn.execute("DELETE FROM password_resets WHERE user_id = ? AND used_at IS NULL", (user_id,))  # one live token per user
            conn.execute("INSERT INTO password_resets (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                         (token_hash, user_id, now.isoformat(), (now + timedelta(seconds=ttl_seconds)).isoformat()))

    def peek(self, token_hash: str) -> str | None:
        """The user id for a live, unused token, WITHOUT consuming it — else None. Check this before validating a
        new password, so a rejected (e.g. too-weak) password does not burn an otherwise-valid token."""
        now = datetime.now(timezone.utc).isoformat()
        with self.db.connect() as conn:
            row = conn.execute("SELECT user_id, expires_at, used_at FROM password_resets WHERE token_hash = ?", (token_hash,)).fetchone()
        if row is None or row["used_at"] is not None or row["expires_at"] < now:
            return None
        return row["user_id"]

    def consume(self, token_hash: str) -> str | None:
        """The user id for a live, unused token, marking it used — else None (unknown, expired, or already used)."""
        now = datetime.now(timezone.utc).isoformat()
        with self.db.connect() as conn:
            row = conn.execute("SELECT user_id, expires_at, used_at FROM password_resets WHERE token_hash = ?", (token_hash,)).fetchone()
            if row is None or row["used_at"] is not None or row["expires_at"] < now:
                return None
            conn.execute("UPDATE password_resets SET used_at = ? WHERE token_hash = ?", (now, token_hash))
            return row["user_id"]
