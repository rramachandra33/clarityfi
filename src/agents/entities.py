"""Deterministic entity extraction for market questions (no LLM): tickers, well-known names, intent."""
from __future__ import annotations

import re

from src.data.news_client import _NOT_TICKERS
from src.utils.validators import normalize_ticker

MAX_SYMBOLS = 8

# Everyday names -> tickers. Index/market phrases map to a widely-held ETF proxy, and the agent says so.
NAME_ALIASES: dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL", "amazon": "AMZN", "tesla": "TSLA",
    "nvidia": "NVDA", "meta": "META", "facebook": "META", "netflix": "NFLX", "berkshire": "BRK-B", "disney": "DIS",
    "coca-cola": "KO", "coca cola": "KO", "walmart": "WMT", "jpmorgan": "JPM", "visa": "V", "mastercard": "MA",
    "bitcoin": "BTC-USD", "ethereum": "ETH-USD", "ether": "ETH-USD",
    "s&p 500": "SPY", "s&p500": "SPY", "sp500": "SPY", "s&p": "SPY", "nasdaq": "QQQ", "nasdaq-100": "QQQ", "dow jones": "DIA",
    "the dow": "DIA", "total stock market": "VTI", "total market": "VTI", "bond market": "BND", "international stocks": "VXUS",
    "gold": "GLD", "real estate": "VNQ", "reits": "VNQ",
}
PROXY_NOTES: dict[str, str] = {
    "SPY": "S&P 500 (shown via the SPY fund)", "QQQ": "Nasdaq-100 (shown via the QQQ fund)", "DIA": "Dow Jones (shown via the DIA fund)",
    "VTI": "total U.S. stock market (shown via the VTI fund)", "BND": "U.S. bond market (shown via the BND fund)",
    "VXUS": "international stocks (shown via the VXUS fund)", "GLD": "gold (shown via the GLD fund)",
    "VNQ": "U.S. real estate (shown via the VNQ fund)",
}
_ALIAS_RE = re.compile(r"(?<![\w-])(" + "|".join(re.escape(k) for k in sorted(NAME_ALIASES, key=len, reverse=True)) + r")(?![\w-])", re.I)
_DOLLAR_TICKER = re.compile(r"\$([A-Za-z]{1,5}(?:[.\-][A-Za-z]{1,4})?)\b")
_CAPS_TICKER = re.compile(r"(?<![\w$-])([A-Z]{2,5}(?:[.\-][A-Z]{1,4})?)(?![\w-])")
_CAPS_STOPWORDS = frozenset({"THE", "AND", "FOR", "ARE", "NOT", "BUT", "HOW", "WHY", "WHAT", "WHEN", "WHO", "CAN", "YOU", "ALL",
                             "ANY", "HAS", "HAD", "WAS", "OUR", "OUT", "NOW", "NEW", "TOP", "USD", "EUR", "GBP", "PE", "OK", "AM",
                             "PM", "TV", "ID", "DIY", "FAQ", "ASAP", "IMO", "FYI", "TLDR", "AI", "IS", "UP", "IT", "IN", "ON", "OR", "AT",
                             "AS", "BE", "DO", "GO", "IF", "MY", "OF", "SO", "TO", "WE", "BY", "HE", "ME", "IS", "NO", "YES",
                             "MARKET", "STOCK"}) | _NOT_TICKERS

_TREND_WORDS = re.compile(r"\b(trend|trending|history|historical|performance|performed|perform|past|over the (last|past)|chart|"
                          r"moving average|volatil\w*|drawdown|how has|how have|year to date|ytd|last (month|year|quarter)|"
                          r"3[- ]?month|1[- ]?month|6[- ]?month)\b", re.I)


def extract_symbols(text: str) -> list[str]:
    """Tickers mentioned in `text`, in order of appearance, deduplicated (max 8).

    Recognizes $TSLA, ALL-CAPS tokens like AAPL / BRK-B / BTC-USD, and well-known names ("Apple",
    "bitcoin", "S&P 500"). Plain lowercase words are never treated as tickers.
    """
    found: list[tuple[int, str]] = []
    for m in _DOLLAR_TICKER.finditer(text):
        sym = normalize_ticker(m.group(1))
        if sym:
            found.append((m.start(), sym))
    words = re.findall(r"[A-Za-z]{2,}", text)
    shouting = len(words) >= 3 and all(w.isupper() for w in words)  # ALL CAPS message: caps carry no ticker signal
    for m in ([] if shouting else _CAPS_TICKER.finditer(text)):
        token = m.group(1)
        if token in _CAPS_STOPWORDS:
            continue
        sym = normalize_ticker(token)
        if sym:
            found.append((m.start(), sym))
    for m in _ALIAS_RE.finditer(text):
        found.append((m.start(), NAME_ALIASES[m.group(1).lower()]))
    ordered: list[str] = []
    for _, sym in sorted(found):
        if sym not in ordered:
            ordered.append(sym)
    return ordered[:MAX_SYMBOLS]


def wants_trend(text: str) -> bool:
    return bool(_TREND_WORDS.search(text))


def proxy_notes(symbols: list[str], text: str) -> list[str]:
    """Explain when a market/index phrase was answered with a fund proxy."""
    notes = []
    lowered = text.lower()
    for phrase, sym in NAME_ALIASES.items():
        if sym in symbols and sym in PROXY_NOTES and phrase in lowered and PROXY_NOTES[sym] not in notes:
            notes.append(PROXY_NOTES[sym])
    return notes
