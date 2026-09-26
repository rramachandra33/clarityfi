"""Input validation helpers shared by the input guard and the API."""
from __future__ import annotations

import re
import unicodedata

from src.core.errors import GuardrailError

_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,4})?$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Heuristics only: flagged input is still answered (the system prompt treats user text as data),
# but the workflow can log/trace the flag and skip tools. Not a security boundary on its own.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
        r"disregard (all |any )?(the )?(previous|prior|above|earlier|your) (instructions|prompts?|rules)",
        r"(reveal|show|print|repeat|output) (me )?(your|the) (system|hidden|initial) (prompt|instructions|message)",
        r"you are now (?:in )?(?:developer|dan|jailbreak|unrestricted)",
        r"\bact as (?:if you (?:have|had) no|an? unrestricted)",
        r"pretend (?:that )?you (?:have no|are not bound|don'?t have) (?:rules|restrictions|guidelines)",
        r"</?\s*(system|assistant)\s*>",
        r"(api|secret)[ _-]?key\s*(?:is|=|:)",
    )
]


def validate_user_input(text: object, max_chars: int = 4000) -> str:
    """Return cleaned text or raise GuardrailError (non-string, empty, or too long)."""
    if not isinstance(text, str):
        raise GuardrailError("Please enter your question as text.")
    cleaned = _CONTROL_RE.sub("", unicodedata.normalize("NFKC", text)).strip()
    if not cleaned:
        raise GuardrailError("Please enter a question.")
    if len(cleaned) > max_chars:
        raise GuardrailError(f"That message is too long (limit {max_chars} characters).")
    return cleaned


def looks_like_prompt_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def normalize_ticker(symbol: object) -> str | None:
    """Uppercase a ticker such as 'brk-b' or 'VTI'; None if it is not ticker-shaped."""
    if not isinstance(symbol, str):
        return None
    candidate = symbol.strip().lstrip("$").upper()
    return candidate if _TICKER_RE.match(candidate) else None
