import pytest

from src.agents.entities import MAX_SYMBOLS, extract_symbols, proxy_notes, wants_trend


@pytest.mark.parametrize("text,expected", [
    ("What is AAPL trading at?", ["AAPL"]),
    ("Compare MSFT, GOOGL and $tsla", ["MSFT", "GOOGL", "TSLA"]),
    ("how is Apple doing versus Microsoft", ["AAPL", "MSFT"]),
    ("Is bitcoin up today?", ["BTC-USD"]),
    ("price of BTC-USD and ETH-USD", ["BTC-USD", "ETH-USD"]),
    ("How is the S&P 500 doing?", ["SPY"]),
    ("what about the Nasdaq and the Dow Jones", ["QQQ", "DIA"]),
    ("thoughts on BRK-B", ["BRK-B"]),
    ("$vti vs VOO", ["VTI", "VOO"]),
    ("AAPL and Apple and $aapl", ["AAPL"]),                     # deduplicated
    ("What is an ETF and how does an IRA work?", []),            # finance jargon in caps is not a ticker
    ("THE MARKET IS UP", []),                                    # shouting is not tickers
    ("I want to buy some apple pie", ["AAPL"]),                  # known limitation: name match without context
    ("", []), ("no symbols here at all", []),
    ("tell me about meta and gold prices", ["META", "GLD"]),
])
def test_extract_symbols(text, expected):
    assert extract_symbols(text) == expected


def test_order_of_appearance_and_cap():
    assert extract_symbols("MSFT then AAPL then NVDA") == ["MSFT", "AAPL", "NVDA"]
    many = "compare " + " ".join(f"T{chr(65 + i)}{chr(65 + i)}" for i in range(15))
    assert len(extract_symbols(many)) == MAX_SYMBOLS


def test_lowercase_words_are_never_tickers():
    assert extract_symbols("aapl msft tsla") == []


@pytest.mark.parametrize("text,expected", [
    ("how has AAPL performed over the last year", True), ("show me the trend for VTI", True), ("volatility of BTC-USD", True),
    ("year to date for SPY", True), ("what is AAPL at", False), ("current price of MSFT", False),
])
def test_wants_trend(text, expected):
    assert wants_trend(text) is expected


def test_proxy_notes_explain_index_phrases_only():
    assert proxy_notes(["SPY"], "how is the S&P 500 doing") == ["S&P 500 (shown via the SPY fund)"]
    assert proxy_notes(["SPY"], "how is SPY doing") == []            # user asked for the fund itself
    assert proxy_notes(["AAPL"], "how is Apple") == []
