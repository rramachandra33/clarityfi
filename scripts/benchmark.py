"""Live benchmark against the real services (uses your API keys; costs a few cents).

    python -m scripts.benchmark

Reports: retrieval quality (is the expected category in the top results?), answer latency cold vs. exact-cache vs.
semantic-cache, and the Q&A cache hit rate. Uses a throwaway user id and conversation ids.
"""
from __future__ import annotations

import statistics
import time

from fastapi.testclient import TestClient

from src.web_app.api import main

# (question, categories that count as a correct source)
RETRIEVAL_SET = [
    ("What is an ETF?", {"etfs_and_funds", "funds", "fund_selection", "emerging_products"}),
    ("How do bonds work?", {"bonds"}),
    ("Why should I diversify my investments?", {"portfolio_management", "portfolio_construction", "risk", "risk_and_return", "investing_basics", "optimization"}),
    ("What is the difference between a Roth and a traditional IRA?", {"retirement", "taxes_and_investing", "tax_efficiency", "tax_education"}),
    ("How are capital gains taxed?", {"taxes_and_investing", "tax_efficiency", "tax_education"}),
    ("What is an emergency fund?", {"personal_finance", "saving_and_investing", "financial_basics", "planning", "cash_and_cash_equivalents", "banking"}),
    ("What does the Sharpe ratio measure?", {"risk_adjusted_performance", "risk_measurement", "portfolio_management"}),
    ("How can I spot investment fraud?", {"fraud_types", "fraud_prevention", "account_protection", "investor_remedies"}),
    ("What is compound interest?", {"financial_basics", "investing_basics", "financial_foundations", "foundations", "saving_and_investing", "personal_finance"}),
    ("What is an expense ratio?", {"costs", "etfs_and_funds", "funds", "fund_selection", "fees_and_conflicts"}),
]
PARAPHRASES = [("What is an ETF?", "Can you explain what an ETF is?"), ("How do bonds work?", "Explain how bonds work please")]


def timed(client, question, conversation):
    start = time.perf_counter()
    response = client.post("/api/chat", json={"user_id": "benchmark-user", "conversation_id": conversation, "question": question})
    return response, (time.perf_counter() - start) * 1000


def main_report() -> None:
    hits = 0
    print("Retrieval quality (top-3 contains an expected category):")
    for question, expected in RETRIEVAL_SET:
        results = main.store.search(question, 3)
        ok = any(str(r.metadata.get("category")) in expected for r in results)
        hits += ok
        print(f"  {'OK ' if ok else 'MISS'} {question}  -> {[r.metadata.get('category') for r in results]}")
    print(f"  => {hits}/{len(RETRIEVAL_SET)} = {hits / len(RETRIEVAL_SET):.0%}\n")

    with TestClient(main.app) as client:
        main.api_services.limiters["chat"].max_calls = 10_000  # do not let the limiter skew the timing
        cold, exact, semantic = [], [], []
        outcomes = {"exact": 0, "semantic": 0, "miss": 0}
        for question, paraphrase in PARAPHRASES:
            main.qa_cache.clear() if hasattr(main.qa_cache, "clear") else None
            r, ms = timed(client, question, "bench-a")
            cold.append(ms) if r.json().get("cache_hit") is None else None
            outcomes["miss" if r.json().get("cache_hit") is None else r.json()["cache_hit"]] += 1
            r, ms = timed(client, question, "bench-b")
            outcomes[r.json().get("cache_hit") or "miss"] += 1
            exact.append(ms) if r.json().get("cache_hit") == "exact" else None
            r, ms = timed(client, paraphrase, "bench-c")
            outcomes[r.json().get("cache_hit") or "miss"] += 1
            semantic.append(ms) if r.json().get("cache_hit") == "semantic" else None
        med = lambda xs: f"{statistics.median(xs):.0f} ms (n={len(xs)})" if xs else "n/a"  # noqa: E731
        print(f"Latency  cold: {med(cold)} | exact cache: {med(exact)} | semantic cache: {med(semantic)}")
        total = sum(outcomes.values())
        print(f"Cache outcomes over {total} requests: {outcomes}  (hit rate {(outcomes['exact'] + outcomes['semantic']) / total:.0%})")
        portfolio_ms = [timed(client, "How is my portfolio doing?", "bench-p")[1] for _ in range(1)]
        print(f"Portfolio question (no LLM cache): {portfolio_ms[0]:.0f} ms")


if __name__ == "__main__":
    main_report()
