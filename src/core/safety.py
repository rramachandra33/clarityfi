"""Harmful-content guard: a safety net in front of the finance assistant, upstream of the router.

Heuristics only (regex pattern matching), like the prompt-injection check in `utils/validators.py` - not a
substitute for a dedicated safety classifier, and can both over- and under-match. Two categories:

- self_harm: a compassionate, deterministic response with real crisis resources. Never sent to an LLM or
  agent - the response itself is generated here, by code, so it cannot be talked around by prompt
  injection and never depends on model behavior. The workflow skips PII redaction and the usual
  disclaimer footer for this response (see `graph.py`'s `skip_output_guard`), so a crisis-line number is
  never mangled by the PII filter and the message doesn't end on an incongruous legal disclaimer.
- violence: a flat refusal for requests for weapons, explosives, or instructions to harm people. Out of
  scope for a finance-education assistant regardless of topic.

Checked in the guard node, before the router and before any agent or LLM call.
"""
from __future__ import annotations

import re

_SELF_HARM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(kill|end)\s+(myself|my own life|my life)\b",
        r"\bcommit(?:ting|ted)?\s+suicide\b",
        r"\bsuicidal\b",
        r"\b(want|wanted|wanting|going)\s+to\s+die\b",
        r"\bself[- ]harm(?:ing)?\b",
        r"\bhurt(?:ing)?\s+myself\b",
        r"\bways?\s+to\s+(kill myself|end it all|end my life|take my (?:own )?life)\b",
        r"\bno\s+reason\s+to\s+live\b",
        r"\boverdose\s+(?:myself|on purpose)\b",
        r"\bhow\s+(?:do|can|to)\s+i\s+(?:kill myself|end my life|take my (?:own )?life)\b",
    )
]

_VIOLENCE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bhow\s+(?:do|to|can)\s+(?:i|you)?\s*(?:make|build|create)\s+an?\s*(?:bomb|pipe bomb|explosive)\b",
        r"\bmake\s+an?\s+(?:bomb|explosive|nerve agent|chemical weapon|biological weapon)\b",
        r"\bhow\s+to\s+(?:make|build|create|synthesize)\s+an?\s*(?:bomb|explosive|poison|nerve agent|chemical weapon)\b",
        r"\bhow\s+(?:do|to|can)\s+(?:i|you)?\s*(?:kill|murder)\s+(?:someone|somebody|a person|my \w+)\b",
        r"\bbuild\s+an?\s+(?:untraceable\s+)?(?:gun|firearm|weapon)\b",
    )
]

SELF_HARM_RESPONSE = (
    "I'm really sorry you're going through this, and I want to take it seriously rather than brush past it. "
    "I'm not able to help with that, but please reach out to people who can:\n\n"
    "- In the US: call or text 988 (Suicide & Crisis Lifeline), available 24/7\n"
    "- Crisis Text Line: text HOME to 741741\n"
    "- Outside the US: findahelpline.com lists local crisis lines\n"
    "- If you're in immediate danger, please call your local emergency number (911 in the US)\n\n"
    "If you'd like, I'm still here for finance and money questions whenever you're ready."
)

HARMFUL_CONTENT_RESPONSE = (
    "I can't help with that. I'm a personal finance and investing education assistant, and I won't provide "
    "instructions for weapons, explosives, or harming people regardless of the topic. Educational only."
)


def detect_harmful_content(text: str) -> str | None:
    """'self_harm', 'violence', or None. A message that matches both is treated as a crisis first."""
    if any(p.search(text) for p in _SELF_HARM_PATTERNS):
        return "self_harm"
    if any(p.search(text) for p in _VIOLENCE_PATTERNS):
        return "violence"
    return None
