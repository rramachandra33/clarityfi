"""Short-term conversation memory (per user + conversation), kept as redacted text only.

Persistence to SQLite and LangGraph checkpointing are added with the workflow (Step 11); this class
is the single seam through which the API reads/writes chat history, so that change stays local.
"""
from __future__ import annotations

import threading
from collections import OrderedDict, deque


class ConversationMemory:
    def __init__(self, max_messages: int = 8, max_conversations: int = 1000) -> None:
        self.max_messages = max_messages
        self.max_conversations = max_conversations
        self._data: OrderedDict[str, deque[dict[str, str]]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(user_id: str, conversation_id: str) -> str:
        return f"{user_id.strip().lower()}:{conversation_id.strip()}"

    def history(self, key: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._data.get(key, ()))

    def append(self, key: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"} or not content.strip():
            return
        with self._lock:
            turns = self._data.setdefault(key, deque(maxlen=self.max_messages))
            turns.append({"role": role, "content": content})
            self._data.move_to_end(key)
            while len(self._data) > self.max_conversations:  # bound memory: drop the least recently active
                self._data.popitem(last=False)

    def record_exchange(self, key: str, question: str, answer: str, **_metadata) -> bool:
        """Same signature as SqliteConversationMemory (agent/citations/cache_hit are accepted and ignored)."""
        self.append(key, "user", question)
        self.append(key, "assistant", answer)
        return True

    def clear(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._data.clear()
            else:
                self._data.pop(key, None)


class SqliteConversationMemory:
    """Conversation memory persisted in the app database (survives restarts), redacted text only.

    Same interface as `ConversationMemory`, so the workflow is agnostic. One session row per
    (user, conversation); messages are stored with the agent that produced them and their citations.
    """

    def __init__(self, db, max_messages: int = 8) -> None:
        from src.data.repositories import MessageRepository, SessionRepository, UserRepository

        self.db, self.max_messages = db, max_messages
        self._users, self._sessions, self._messages = UserRepository(db), SessionRepository(db), MessageRepository(db)

    @staticmethod
    def key(user_id: str, conversation_id: str) -> str:
        return f"{user_id.strip()}:{conversation_id.strip()}"

    @staticmethod
    def _split(key: str) -> tuple[str, str]:
        user_id, _, conversation = key.partition(":")  # user ids cannot contain ':'
        return user_id, conversation

    def history(self, key: str) -> list[dict[str, str]]:
        user_id, conversation = self._split(key)
        profile = self._users.get(user_id)
        if profile is None:
            return []
        session_id = self._sessions.find(profile.user_id, conversation)
        if session_id is None:
            return []
        rows = self._messages.history(session_id, limit=self.max_messages)
        return [{"role": r["role"], "content": r["content"]} for r in rows if r["role"] in ("user", "assistant")]

    def record_exchange(self, key: str, question: str, answer: str, *, agent: str | None = None,
                        citations: list[dict] | None = None, cache_hit: str | None = None) -> bool:
        """Persist one Q/A turn. Returns False (nothing stored) if the user id is not storable."""
        from pydantic import ValidationError

        if not question.strip() or not answer.strip():
            return False
        user_id, conversation = self._split(key)
        try:
            profile, _created = self._users.ensure(user_id)
        except ValidationError:
            return False
        session_id = self._sessions.get_or_create(profile.user_id, conversation)
        self._messages.add(session_id, "user", question)
        self._messages.add(session_id, "assistant", answer, agent=agent, citations=citations, cache_hit=cache_hit)
        return True

    def clear(self, key: str | None = None) -> None:
        if key is None:
            with self.db.connect() as conn:
                conn.execute("DELETE FROM messages")
            return
        user_id, conversation = self._split(key)
        profile = self._users.get(user_id)
        session_id = self._sessions.find(profile.user_id, conversation) if profile else None
        if session_id:
            with self.db.connect() as conn:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
