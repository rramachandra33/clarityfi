import json

import pytest

from src.agents.base import AgentRequest
from src.agents.portfolio import NEEDS_PORTFOLIO, PortfolioAgent
from src.core.errors import LLMError
from src.core.freshness import DataStatus
from src.core.models import Portfolio
from src.core.portfolio_calc import compute_portfolio_metrics
from src.data.db import Database
from src.data.repositories import PortfolioRepository
from src.data.seed import seed
from tests.unit._agent_fakes import FakeLLM, FakeMarket, quote

PRICES = {"VTI": 300.0, "VXUS": 60.0, "BND": 75.0}


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "pf.db")
    d.init_schema()
    seed(d)
    return d


def live_market():
    return FakeMarket({t: quote(t, p) for t, p in PRICES.items()})


def agent(db, market=None, llm=None):
    llm = llm or FakeLLM()
    a = PortfolioAgent(db, market or live_market(), llm_factory=lambda: llm, callbacks_factory=lambda: [])
    a.llm = llm
    return a


def go(a, user="learner-105", question="How is my portfolio doing?", **kw):
    return a.run(AgentRequest(question=question, user_id=user, **kw))


def expected_metrics(db, user="learner-105", prices=PRICES):
    from src.data.repositories import UserRepository
    profile = UserRepository(db).get(user)
    return compute_portfolio_metrics(PortfolioRepository(db).get(user), prices, stated_risk=profile.risk_tolerance)


def facts_sent(a):
    return json.loads(a.llm.messages[0][-1].content.split("FACTS:\n", 1)[1])


def test_three_fund_portfolio_is_valued_and_scored_by_the_calculator(db):
    m = expected_metrics(db)
    llm = FakeLLM(f"Your portfolio is worth ${m.total_value:,.2f}, and diversification scores {m.diversification_score:.1f} out of 100.")
    a = agent(db, llm=llm)
    r = go(a)
    assert r.ok and r.data["narration"] == "llm" and r.content.startswith("Your portfolio is worth")
    assert r.data["metrics"]["total_value"] == m.total_value and r.data["metrics"]["diversification_score"] == m.diversification_score
    assert r.data["priced_with"] == ["BND", "VTI", "VXUS"] and r.data["unpriced"] == [] and r.notice is None
    f = facts_sent(a)
    assert f["total_value"] == m.total_value and f["risk_label"] == m.risk_label and f["stated_risk_tolerance"] == "moderate"
    assert "how_scores_work" in f and f["data_notices"] == []


def test_holdings_table_is_sorted_and_weighted_for_the_dashboard(db):
    r = go(agent(db))
    rows = r.data["holdings"]
    assert [x["ticker"] for x in rows] == ["VTI", "VXUS", "BND"]  # largest value first
    assert rows[0]["price"] == 300.0 and rows[0]["value"] == 260 * 300.0 and rows[0]["price_source"] == "Alpha Vantage"
    assert sum(x["weight_pct"] for x in rows) < 100.01 and rows[0]["price_status"] == "live"


def test_empty_portfolio_and_unknown_user_ask_for_holdings_without_any_calls(db):
    m = live_market()
    a = agent(db, m)
    for user in ("learner-110", "brand-new-user"):
        r = go(a, user=user)
        assert r.content == NEEDS_PORTFOLIO and r.data["needs_portfolio"] is True
    assert m.quote_calls == [] and a.llm.calls == 0


def test_cash_only_portfolio_needs_no_market_lookup(db):
    m = live_market()
    a = agent(db, m, FakeLLM("You hold $24,000.00 in cash, so nothing is invested."))
    r = go(a, user="learner-103")
    assert r.ok and m.quote_calls == [] and r.data["metrics"]["risk_label"] == "Conservative" and r.data["holdings"] == []
    assert r.data["metrics"]["total_value"] == 24000.0


def test_cached_prices_are_disclosed_and_set_the_status(db):
    market = FakeMarket({t: quote(t, p, status=DataStatus.cached, age=1200) for t, p in PRICES.items()})
    r = go(agent(db, market))
    assert r.data_status is DataStatus.cached and "Cached data from Alpha Vantage" in r.notice and "20 minutes" in r.notice
    assert "Data note:" in r.content and r.data["notices"] and len(r.data["notices"]) == 1  # deduplicated across holdings


def test_sample_prices_are_never_used_to_value_a_portfolio(db):
    market = FakeMarket({"VTI": quote("VTI", 1.0, status=DataStatus.sample, source="mock"), "VXUS": quote("VXUS", 60.0),
                         "BND": quote("BND", 75.0)})
    r = go(agent(db, market))
    assert r.data["sample_quotes_ignored"] == ["VTI"] and r.data["unpriced"] == ["VTI"]
    assert r.data_status is DataStatus.stale and "valued at your cost basis" in r.notice and "VTI" in r.notice
    vti = next(x for x in r.data["holdings"] if x["ticker"] == "VTI")
    assert vti["price"] is None and vti["price_source"] == "cost basis" and vti["value"] == 52000.0  # seeded cost basis, not $1 x shares


def test_holdings_the_market_cannot_price_fall_back_to_cost_basis_with_a_notice(db):
    r = go(agent(db, FakeMarket({"VTI": quote("VTI", 300.0)})))
    assert r.data["unpriced"] == ["BND", "VXUS"] and "BND, VXUS" in r.notice and r.data_status is DataStatus.stale
    assert r.data["errors"].keys() == {"VXUS", "BND"}


def test_worst_status_wins_across_holdings(db):
    market = FakeMarket({"VTI": quote("VTI", 300.0), "VXUS": quote("VXUS", 60.0, status=DataStatus.stale, age=3 * 3600),
                         "BND": quote("BND", 75.0, status=DataStatus.cached, age=600)})
    assert go(agent(db, market)).data_status is DataStatus.stale


def test_context_portfolio_overrides_the_saved_one_and_is_analyzed(db):
    ctx = {"portfolio": {"holdings": [{"ticker": "AAPL", "shares": 10, "asset_class": "stock"}], "cash": 0}}
    market = FakeMarket({"AAPL": quote("AAPL", 336.13)})
    a = agent(db, market, FakeLLM("All of it, $3,361.30, is in one stock."))
    r = go(a, context=ctx)
    assert market.quote_calls == [["AAPL"]] and r.data["metrics"]["total_value"] == 3361.3
    assert any("makes up 100%" in n for n in r.data["metrics"]["notes"])


def test_context_may_pass_a_portfolio_object(db):
    r = go(agent(db, FakeMarket({})), context={"portfolio": Portfolio(user_id="x", cash=500)})
    assert r.data["metrics"]["total_value"] == 500.0


def test_stated_risk_tolerance_creates_alignment_observations(db):
    # learner-104 is a concentrated single-stock investor with an aggressive stated tolerance
    market = FakeMarket({"AAPL": quote("AAPL", 118.0), "VOO": quote("VOO", 400.0)})
    a = agent(db, market)
    go(a, user="learner-104")
    assert facts_sent(a)["stated_risk_tolerance"] == "aggressive" and any("AAPL makes up" in n for n in facts_sent(a)["observations"])


def test_llm_outage_returns_a_complete_deterministic_analysis(db):
    r = go(agent(db, llm=FakeLLM(error=LLMError("down"))))
    m = expected_metrics(db)
    assert r.ok and r.data["narration"] == "template"
    assert f"${m.total_value:,.2f}" in r.content and f"Diversification score: {m.diversification_score:.0f} out of 100" in r.content
    assert "Largest holdings: VTI" in r.content and "Educational only" in r.content


def test_invented_figures_in_narration_are_rejected(db):
    r = go(agent(db, llm=FakeLLM("Your portfolio is worth $999,999 and you should sell everything.")))
    assert r.data["narration"] == "template" and "999,999" not in r.content


def test_prompt_contains_no_holdings_beyond_computed_facts_and_question_is_included(db):
    a = agent(db)
    go(a, question="Is my portfolio too risky?")
    human = a.llm.messages[0][-1].content
    assert "Is my portfolio too risky?" in human and "learner-105" not in human  # user id is never sent to the LLM


def test_every_synthetic_user_can_be_analyzed_end_to_end(db):
    full = FakeMarket({t: quote(t, 100.0) for t in ["QQQ", "VTI", "NVDA", "TSLA", "BND", "SCHD", "TIP", "AAPL", "VOO", "VXUS", "AGG",
                                                    "VGSH", "BTC-USD", "ETH-USD", "COIN", "MSFT", "VNQ", "O"]})
    for i in range(101, 111):
        r = go(agent(db, full), user=f"learner-{i}")
        assert r.ok, (i, r.error)
        if i == 110:
            assert r.data["needs_portfolio"]
        else:
            assert 0 <= r.data["metrics"]["diversification_score"] <= 100 and r.data["metrics"]["total_value"] > 0
