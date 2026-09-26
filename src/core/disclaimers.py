"""Education-not-advice language shared by every agent and the output guard."""
from __future__ import annotations

EDUCATIONAL_DISCLAIMER = (
    "Educational information only - not personalized financial, investment, legal, or tax advice. "
    "Investing involves risk, including loss of principal."
)

MARKET_DATA_DISCLAIMER = "Market data may be delayed or cached and is shown for learning purposes."

ADVICE_REDIRECT = (
    "I can't tell you what to buy or sell, but I can explain the concepts that help you decide: "
    "how the investment works, its risks and costs, and how it fits a goal and time horizon. "
    "For personal recommendations, consider a licensed financial professional."
)


def with_disclaimer(text: str, *, market_data: bool = False) -> str:
    """Append the standard disclaimer(s) once (idempotent)."""
    footer = EDUCATIONAL_DISCLAIMER
    if market_data:
        footer = f"{MARKET_DATA_DISCLAIMER} {footer}"
    if footer in text:
        return text
    return f"{text.rstrip()}\n\n_{footer}_"


def with_data_notices(text: str, notices: list[str] | None) -> str:
    """Append data-provenance notices (cached / last-known / sample) so an answer never hides them.

    Plain text (no markdown) so it reads correctly in the UI and when spoken. Agents call this on any
    narrated answer built from market or news data.
    """
    unique = [n for n in dict.fromkeys(notices or []) if n]
    if not unique:
        return text
    block = "\n".join(f"- {n}" for n in unique)
    footer = f"\n\nData note:\n{block}"
    return text if block in text else text.rstrip() + footer


_DISCLAIMER_MARKERS = ("educational only", "educational information only", "not personalized financial")
PLAIN_DISCLAIMER = "Educational only: this is general information, not personalized financial, investment, legal, or tax advice."


def ensure_educational_disclaimer(text: str) -> str:
    """Guarantee the answer carries an education-not-advice statement (plain text, safe for voice and UI)."""
    lowered = text.lower()
    if any(marker in lowered for marker in _DISCLAIMER_MARKERS):
        return text
    return f"{text.rstrip()}\n\n{PLAIN_DISCLAIMER}"
