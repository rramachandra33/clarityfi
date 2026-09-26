import json

import pytest

from src.agents.base import AgentRequest
from src.agents.market import ASK_FOR_SYMBOLS, MarketAgent
from src.core.errors import LLMError
from src.core.freshness import DataStatus
from tests.unit._agent_fakes import FakeLLM, FakeMarket, quote


def agent(market, llm=None):
    llm = llm or FakeLLM("AAPL is at $336.13, down 0.26% from the previous close.")
    a = MarketAgent(market, llm_factory=lambda: llm, callbacks_factory=lambda: [])
    a.llm = llm
    return a


def go(a, question, **kw):
    return a.run(AgentRequest(question=question, **kw))


def facts_sent(a):
    human = a.llm.messages[0][-1].content
    return json.loads(human.split("FACTS:\n", 1)[1])


AAPL = quote("AAPL", 336.13, change=-0.2582)


def test_live_quote_is_explained_with_grounded_numbers_and_no_notice():
    a = agent(FakeMarket({"AAPL": AAPL}))
    r = go(a, "What is AAPL trading at?")
    assert r.ok and r.content.startswith("AAPL is at $336.13") and r.notice is None and r.data_status is DataStatus.live
    assert r.data["symbols"] == ["AAPL"] and r.data["quotes"][0]["price"] == 336.13 and r.data["narration"] == "llm"
    f = facts_sent(a)
    assert f["quotes"][0]["price"] == 336.13 and f["quotes"][0]["source"] == "Alpha Vantage" and f["data_notices"] == []
    assert f["quotes"][0]["price_is_as_of_trading_day"] == "2026-09-18" and "previous_session_close" in f["quotes"][0]
    assert "as_of_utc" not in f  # today's date must not tempt the model into "as of today"
    assert "Data note" not in r.content


def test_no_symbol_asks_which_one_without_calling_anything():
    m = FakeMarket({})
    a = agent(m)
    r = go(a, "How is the market doing today?")
    assert r.content == ASK_FOR_SYMBOLS and r.data == {"needs_symbols": True, "latency_ms": r.data["latency_ms"]}
    assert m.quote_calls == [] and a.llm.calls == 0


def test_names_and_index_phrases_are_resolved_and_proxies_are_explained():
    m = FakeMarket({"SPY": quote("SPY", 761.69), "AAPL": AAPL})
    a = agent(m, FakeLLM("SPY is $761.69 and AAPL is $336.13."))
    r = go(a, "How are Apple and the S&P 500 doing?")
    assert m.quote_calls == [["AAPL", "SPY"]] or m.quote_calls == [["SPY", "AAPL"]]
    assert "S&P 500 (shown via the SPY fund)" in facts_sent(a)["index_proxies"]


def test_context_symbols_override_and_are_validated():
    m = FakeMarket({"VTI": quote("VTI", 379.98)})
    a = agent(m, FakeLLM("VTI is $379.98."))
    r = go(a, "how is it", context={"symbols": ["vti", "bad symbol!", "$vti"]})
    assert m.quote_calls == [["VTI", "VTI"]] and r.data["symbols"] == ["VTI", "VTI"] or r.data["symbols"][0] == "VTI"


# ---------------------------------------------------------------------------- transparency rule
def test_cached_quote_is_disclosed_in_notice_status_and_text():
    q = quote("AAPL", 336.13, status=DataStatus.cached, age=780)
    r = go(agent(FakeMarket({"AAPL": q})), "What is AAPL at?")
    assert r.data_status is DataStatus.cached and "Cached data from Alpha Vantage" in r.notice and "13 minutes" in r.notice
    assert "Data note:" in r.content and "13 minutes" in r.content and "**" not in r.content


def test_sample_quote_is_labelled_sample_everywhere():
    q = quote("VOO", 494.02, status=DataStatus.sample, source="mock")
    r = go(agent(FakeMarket({"VOO": q}), FakeLLM("VOO is $494.02.")), "What is VOO at?")
    assert r.data_status is DataStatus.sample and "SAMPLE" in r.notice and "SAMPLE" in r.content


def test_overall_status_is_the_worst_of_all_quotes_and_notices_are_deduplicated():
    qs = {"AAPL": AAPL, "MSFT": quote("MSFT", 493.46, status=DataStatus.stale, age=3 * 3600),
          "VTI": quote("VTI", 379.98, status=DataStatus.stale, age=3 * 3600)}
    r = go(agent(FakeMarket(qs), FakeLLM("AAPL $336.13, MSFT $493.46, VTI $379.98.")), "AAPL MSFT VTI prices")
    assert r.data_status is DataStatus.stale and len(r.data["notices"]) == 1 and "3 hours" in r.data["notices"][0]


# ----------------------------------------------------------------------------------- unknown symbols
def test_unknown_symbols_are_reported_while_known_ones_are_answered():
    r = go(agent(FakeMarket({"AAPL": AAPL})), "Compare AAPL and ZZZQ")
    assert r.ok and r.data["unknown_symbols"] == ["ZZZQ"] and r.data["symbols"] == ["AAPL", "ZZZQ"]


def test_all_unknown_symbols_gives_a_helpful_message_without_calling_the_llm():
    a = agent(FakeMarket({}))
    r = go(a, "price of ZZZQ")
    assert r.ok and "couldn't find: ZZZQ" in r.content and a.llm.calls == 0 and r.data["errors"] == {"ZZZQ": "Ticker was not found"}


# ------------------------------------------------------------------------------------------ trends
def test_trend_is_included_when_asked_and_limited_to_three_symbols():
    m = FakeMarket({s: quote(s, 100 + i) for i, s in enumerate(["AAA", "BBB", "CCC", "DDD"])})
    a = agent(m, FakeLLM("Trends are shown."))
    r = go(a, "How has AAA, BBB, CCC and DDD performed over the last year?")
    assert m.trend_calls == ["AAA", "BBB", "CCC"] and len(r.data["trends"]) == 3
    assert facts_sent(a)["trends"][0]["label"] == "Uptrend"


def test_trend_can_be_forced_by_context_and_is_skipped_otherwise():
    m = FakeMarket({"AAPL": AAPL})
    go(agent(m), "What is AAPL at?")
    assert m.trend_calls == []
    go(agent(m), "What is AAPL at?", context={"include_trend": True})
    assert m.trend_calls == ["AAPL"]


def test_non_live_trend_data_is_disclosed_and_worsens_status():
    m = FakeMarket({"AAPL": AAPL}, trend_status=DataStatus.stale)
    r = go(agent(m, FakeLLM("AAPL is $336.13.")), "AAPL trend over the past 3 months")
    assert r.data_status is DataStatus.stale and any("price history" in n for n in r.data["notices"])


def test_a_failing_trend_lookup_does_not_lose_the_quote():
    r = go(agent(FakeMarket({"AAPL": AAPL}, trend_fails=True)), "AAPL performance history")
    assert r.ok and r.data["quotes"] and r.data["trends"] == []


# -------------------------------------------------------------------------------- narration safety
def test_llm_outage_falls_back_to_deterministic_text_with_the_real_numbers():
    r = go(agent(FakeMarket({"AAPL": AAPL}), FakeLLM(error=LLMError("down"))), "What is AAPL at?")
    assert r.ok and r.data["narration"] == "template" and "AAPL: $336.13 (-0.26% vs the previous close)" in r.content
    assert "Alpha Vantage" in r.content and "do not predict future results" in r.content


def test_invented_prices_are_rejected_and_replaced_by_the_template():
    r = go(agent(FakeMarket({"AAPL": AAPL}), FakeLLM("AAPL is $410.00 and will hit $500 soon.")), "What is AAPL at?")
    assert r.data["narration"] == "template" and "410" not in r.content and "336.13" in r.content


def test_answer_always_carries_the_education_disclaimer():
    r = go(agent(FakeMarket({"AAPL": AAPL}), FakeLLM("AAPL is at $336.13.")), "What is AAPL at?")
    assert "educational" in r.content.lower()


def test_prompt_tells_the_model_about_trading_days_and_absent_notices():
    a = agent(FakeMarket({"AAPL": AAPL}))
    go(a, "What is AAPL at?")
    system = a.llm.messages[0][0].content
    assert "price_is_as_of_trading_day" in system and "never say 'today'" in system
    assert "If data_notices is empty, say nothing at all about data freshness" in system


def test_template_includes_trends_unknown_symbols_and_index_proxies_when_the_llm_is_down():
    m = FakeMarket({"SPY": quote("SPY", 761.69, change=None)})
    r = go(agent(m, FakeLLM(error=LLMError("down"))), "How has the S&P 500 performed over the last year, and what about ZZZQ?")
    assert r.data["narration"] == "template"
    assert "SPY: $761.69 (source: Alpha Vantage" in r.content            # no change figure available: no parenthetical
    assert "SPY trend over the last 120 trading days: Uptrend." in r.content and "1-month change" in r.content
    assert "3-month change" in r.content and "Annualized volatility" in r.content and "Largest drop from a peak" in r.content
    assert "I couldn't find: ZZZQ." in r.content and "S&P 500 (shown via the SPY fund)" in r.content
