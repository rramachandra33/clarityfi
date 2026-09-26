"""Password hashing and session tokens (standard library only).

- Passwords: scrypt with a random per-password salt; verified in constant time. Format `scrypt$n$r$p$salt$hash`.
- Session tokens: 32 random bytes (URL-safe). Only the SHA-256 of a token is ever stored, so a database leak does not
  hand out usable sessions.
- Unknown users still cost one hash verification (`DUMMY_HASH`) so response time does not reveal which ids exist.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

from src.core.config import get_config
from src.core.errors import GuardrailError

SCRYPT_N = 2 ** 14   # tests lower this for speed; stored hashes carry their own parameters
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LEN = 32
_MAX_MEM = 64 * 1024 * 1024


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=KEY_LEN, maxmem=_MAX_MEM)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str | None) -> bool:
    """True only for a matching password. A missing or malformed hash is verified against a dummy so timing stays similar."""
    try:
        scheme, n, r, p, salt_b64, digest_b64 = (stored or DUMMY_HASH).split("$")
        if scheme != "scrypt":
            raise ValueError
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(digest_b64)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected), maxmem=_MAX_MEM)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected) and stored is not None


def validate_password(password: object) -> str:
    cfg = get_config().auth
    if not isinstance(password, str):
        raise GuardrailError("Enter a password.")
    if len(password) < cfg.min_password_length:
        raise GuardrailError(f"Use a password of at least {cfg.min_password_length} characters.")
    if len(password) > cfg.max_password_length:
        raise GuardrailError(f"Use a password of at most {cfg.max_password_length} characters.")
    if password.strip() != password or any(ord(c) < 32 for c in password):
        raise GuardrailError("The password must not start or end with a space or contain control characters.")
    return password


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


DUMMY_HASH = "scrypt$16384$8$1$" + _b64(b"0" * 16) + "$" + _b64(b"0" * KEY_LEN)
