"""Accounts and sign-in: register, login, logout, and resolving a session cookie to a user.

Threat model (a local/shared demo, not a bank): passwords are scrypt-hashed, tokens are random and stored only as hashes,
failed sign-ins are throttled per client+user id and per client, and error messages do not reveal whether an id exists.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import ValidationError

from src.core import auth
from src.core.config import get_config, settings
from src.core.email import EmailSender, get_email_sender, password_reset_message
from src.core.errors import EmailError, GuardrailError
from src.core.models import UserProfile
from src.data.db import Database
from src.data.repositories import AuthRepository, PasswordResetRepository, UserRepository
from src.web_app.errors import ApiError
from src.web_app.ratelimit import SlidingWindowLimiter

INVALID_LOGIN = "Incorrect user id or password."
RESET_REQUESTED = "If that email is on an account, a reset link has been sent."
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Principal:
    """Who is making the request. `any_user` is only used by tests, to bypass ownership checks."""
    user_id: str
    is_admin: bool = False
    any_user: bool = False

    def owns(self, user_id: str) -> bool:
        return self.any_user or self.user_id.strip().lower() == (user_id or "").strip().lower()


def admin_ids() -> frozenset[str]:
    return frozenset(x.strip().lower() for x in settings.admin_user_ids.split(",") if x.strip())


class AuthService:
    def __init__(self, db: Database, email_sender: EmailSender | None = None) -> None:
        cfg = get_config().auth
        self.cfg = cfg
        self.users, self.repo = UserRepository(db), AuthRepository(db)
        self.resets = PasswordResetRepository(db)
        self.email_sender = email_sender or get_email_sender(settings)
        self.by_user = SlidingWindowLimiter(cfg.login_attempts, cfg.login_window_seconds)
        self.by_client = SlidingWindowLimiter(cfg.ip_login_attempts, cfg.login_window_seconds)
        self.by_reset_client = SlidingWindowLimiter(cfg.reset_requests_per_window, cfg.login_window_seconds)

    # ---- demo accounts -----------------------------------------------------------------------
    def ensure_demo_passwords(self) -> int:
        """Give the synthetic demo users the documented demo password (only where none is set)."""
        if not settings.demo_login_enabled:
            return 0
        done = 0
        for uid in self.repo.users_without_password(synthetic_only=True):
            done += self.repo.set_password_if_unset(uid, auth.hash_password(settings.demo_password))
        return done

    # ---- register / login / logout ---------------------------------------------------------------
    def register(self, user_id: str, password: str, email: str | None = None) -> tuple[UserProfile, str]:
        try:
            uid = UserProfile(user_id=user_id.strip()).user_id
        except ValidationError as exc:
            raise ApiError(422, "User id may use letters, numbers, spaces and . _ @ - (3-120 characters).", "invalid_user_id") from exc
        try:
            auth.validate_password(password)
        except GuardrailError as exc:
            raise ApiError(422, str(exc), "weak_password") from exc
        clean_email = None
        if email and email.strip():
            try:
                clean_email = UserProfile(user_id=uid, email=email).email
            except ValidationError as exc:
                raise ApiError(422, "Enter a valid email address.", "invalid_email") from exc
        existing = self.users.get(uid)
        if existing is not None and (self.repo.get_hash(uid) or self.users.is_synthetic(uid)):
            raise ApiError(409, "That user id is already taken. Sign in instead, or choose another id.", "user_id_taken")
        if existing is None:
            self.users.ensure(uid)
        # An id with no password yet (created by an earlier version of the app) is claimed by the first person to register it.
        if not self.repo.set_password_if_unset(uid, auth.hash_password(password)):
            raise ApiError(409, "That user id is already taken. Sign in instead, or choose another id.", "user_id_taken")
        if clean_email and not self.users.set_email(uid, clean_email):
            raise ApiError(409, "That email is already used by another account.", "email_taken")
        profile = self.users.get(uid)
        return profile, self.start_session(profile.user_id)

    # ---- password reset ------------------------------------------------------------------------
    def request_password_reset(self, email: str, client: str) -> None:
        """Always completes the same way to the caller, whether or not the email matches an account."""
        allowed, retry = self.by_reset_client.peek(client)
        if not allowed:
            raise ApiError(429, f"Too many requests. Please wait {retry} seconds and try again.", "too_many_attempts",
                           headers={"Retry-After": str(retry)})
        self.by_reset_client.hit(client)
        email = (email or "").strip().lower()
        uid = self.users.find_id_by_email(email) if email else None
        if uid is None:
            return
        token = auth.new_token()
        self.resets.create(uid, auth.token_hash(token), self.cfg.reset_token_ttl_seconds)
        reset_url = f"{settings.app_base_url}/?reset_token={token}"
        message = password_reset_message(email, reset_url, self.cfg.reset_token_ttl_seconds // 60)
        try:
            self.email_sender.send(message)
        except EmailError:
            logger.error("Password-reset email failed to send for user_id=%s (not surfaced to the caller)", uid)

    def reset_password(self, token: str, new_password: str) -> bool:
        """False for an unknown, expired, or already-used token. Raises on a weak new password (token stays valid,
        so a typo does not force the user to request a whole new email)."""
        token_hash = auth.token_hash(token or "")
        if self.resets.peek(token_hash) is None:
            return False
        try:
            auth.validate_password(new_password)
        except GuardrailError as exc:
            raise ApiError(422, str(exc), "weak_password") from exc
        uid = self.resets.consume(token_hash)
        if uid is None:  # consumed by a concurrent request between peek and here
            return False
        self.repo.set_password(uid, auth.hash_password(new_password))
        self.repo.delete_user_sessions(uid)  # a reset invalidates every existing session (e.g. if the account was compromised)
        return True

    def login(self, user_id: str, password: str, client: str) -> tuple[UserProfile, str]:
        uid = (user_id or "").strip()
        key = f"{client}|{uid.lower()}"
        for limiter, k in ((self.by_user, key), (self.by_client, client)):
            allowed, retry = limiter.peek(k)
            if not allowed:
                raise ApiError(429, f"Too many sign-in attempts. Please wait {retry} seconds and try again.", "too_many_attempts",
                               headers={"Retry-After": str(retry)})
        stored = self.repo.get_hash(uid) if uid else None
        ok = auth.verify_password(password if isinstance(password, str) else "", stored)  # always runs one verification
        if not (ok and stored):
            self.by_user.hit(key)
            self.by_client.hit(client)
            raise ApiError(401, INVALID_LOGIN, "invalid_credentials")
        profile = self.users.get(uid)
        return profile, self.start_session(profile.user_id)

    def start_session(self, user_id: str) -> str:
        token = auth.new_token()
        self.repo.create_session(user_id, auth.token_hash(token), self.cfg.session_days * 86400)
        return token

    def logout(self, token: str | None) -> None:
        if token:
            self.repo.delete_session(auth.token_hash(token))

    # ---- session cookie -> principal ------------------------------------------------------------------
    def principal_for(self, token: str | None) -> Principal | None:
        if not token:
            return None
        uid = self.repo.user_for_session(auth.token_hash(token))
        return Principal(user_id=uid, is_admin=uid.lower() in admin_ids()) if uid else None
