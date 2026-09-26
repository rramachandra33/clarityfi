"""Plain-text normalization for model output.

The chat UI and voice playback render plain text, but models like to emit markdown. The prompts ask for
plain text; this makes it a guarantee. Only unambiguous markdown is removed (bold, headings, backticks,
asterisk bullets), so ordinary text such as "5 * 3" is left alone.
"""
from __future__ import annotations

import re

_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_UNDERSCORE_BOLD = re.compile(r"(?<!\w)__(.+?)__(?!\w)", re.S)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_BULLET = re.compile(r"^(\s*)[*•]\s+", re.M)
_BACKTICKS = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"^```\w*\s*$", re.M)
_BLANKS = re.compile(r"\n{3,}")


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(])")
_MISSING_INFO = re.compile(
    r"\b(not provided|(is|are) not (included|available|given)|does not (include|contain|provide|mention)|do not (include|contain|provide)|"
    r"no [\w\s\-]{1,40} (is|are|was|were) (provided|available|given|included)|there (is|are) no [\w\s\-]{1,40} (provided|available|given)|"
    r"(i|we) (cannot|can't|am unable to|are unable to|don't have|do not have) [\w\s,'\-]{0,50}(information|data|details|trends?|price)|"
    r"in the facts|facts (do not|don't)|provided facts)\b", re.I)


def drop_missing_info_sentences(text: str) -> str:
    """Remove sentences that complain about missing information ("the price is not provided in the FACTS").

    Data narrations are told never to mention gaps, but models do anyway; the gap usually belongs to a
    different agent handling another part of the question. Paragraphs left empty are removed; if
    everything would be removed the original text is kept.
    """
    kept_paragraphs = []
    for paragraph in text.split("\n\n"):
        lines = []
        for line in paragraph.split("\n"):
            sentences = _SENTENCE.split(line)
            kept = [s for s in sentences if not _MISSING_INFO.search(s)]
            if kept:
                lines.append(" ".join(kept))
        if lines:
            kept_paragraphs.append("\n".join(lines))
    cleaned = "\n\n".join(kept_paragraphs).strip()
    return cleaned or text


def to_plain_text(text: str) -> str:
    out = _FENCE.sub("", text)
    out = _BOLD.sub(r"\1", out)
    out = _UNDERSCORE_BOLD.sub(r"\1", out)
    out = _HEADING.sub("", out)
    out = _BULLET.sub(r"\1- ", out)
    out = _BACKTICKS.sub(r"\1", out)
    return _BLANKS.sub("\n\n", out).strip()
