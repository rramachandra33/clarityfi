# Market Analysis — UI → backend workflow

## Example used

Symbol: **AAPL**, compared against the default benchmark **SPY**.

Source test: `tests/unit/test_market_context.py::test_context_compares_with_the_default_benchmark`

```python
r = client.get("/api/market/context/aapl").json()
assert r["symbol"] == "AAPL" and r["benchmark"] == "SPY" and r["insights"] and r["insights"][-1].startswith("This describes")
assert set(r["comparison"]) == {"AAPL", "SPY"} and r["comparison"]["AAPL"][0]["value"] == 100.0
assert r["relative"]["return_3m_diff_pts"] is not None and client.market.trend_calls == ["AAPL", "SPY"]
```

**Important architectural note:** Market Analysis is the one tab with **no LLM call anywhere in its
normal flow**. Every quote, trend, comparison and "what this may mean" sentence is computed by plain
Python functions. There is a `MarketAgent` (`src/agents/market.py`) that *can* narrate market data with
an LLM, but it is only reachable through the Finance Q&A / Tax chat path (asking "What is AAPL trading
at?" in the chat box), never from this tab.

---

## 1. User action in the browser

The user opens the **Market Analysis** tab, which immediately loads a watchlist, then clicks the **AAPL**
chip (already in the default watchlist `VTI,VXUS,BND,AAPL,MSFT`).

| Step | Function | File |
|---|---|---|
| Tab mounted | `MarketAnalysis()` | `frontend/src/tabs/Market.jsx` |
| Segment overview (auto) | `Overview()` → `useApi('/api/market/overview')` | `frontend/src/tabs/Market.jsx` |
| Watchlist quotes (auto) | `useApi('/api/market/quotes?symbols=...')` | `frontend/src/tabs/Market.jsx` |
| User clicks "AAPL" | `setSelected('AAPL')` | `frontend/src/tabs/Market.jsx` |
| Trend + comparison (auto, on selection) | `useApi('/api/market/trend/AAPL?days=180')` and `useApi('/api/market/context/AAPL?benchmark=SPY')` | `frontend/src/tabs/Market.jsx` |

All four calls go through `useApi()` → `api()` → `apiFetch()` (`frontend/src/api.js`, `frontend/src/http.js`).
This trace follows the `/api/market/context/AAPL` request, since it's the richest one (it also runs the
same code path as `/api/market/trend`).

## 2. Frontend request

```
GET /api/market/context/AAPL?benchmark=SPY&days=180
```

## 3. FastAPI request pipeline

`request_context` middleware → `CORSMiddleware` (**[EXTERNAL]** Starlette) → `require_login()`
(`src/web_app/security.py`) — this endpoint has no `{user_id}` in its path, so ownership checking is
skipped; only "is someone signed in" applies. All in `src/web_app/api/main.py` / `security.py`.

## 4. Route handler

`context(request, symbol, benchmark, days, svc)` in `src/web_app/api/routes_market.py`:

1. `enforce(svc.limiters["read"], client_key(request))` (`src/web_app/ratelimit.py`).
2. `normalize_ticker("aapl")` and `normalize_ticker("SPY")` (`src/utils/validators.py`) → `"AAPL"`, `"SPY"`.
3. `await asyncio.to_thread(svc.market.get_trend, "AAPL", 180)` — the symbol's own history and trend.
4. `await asyncio.to_thread(svc.market.get_trend, "SPY", 180)` — the benchmark's, in the same way (skipped if the symbol IS the benchmark; and, per the test, this call happens even if the first one's `data_status` isn't live — the benchmark is fetched independently).
5. `relative_performance(summary, bench_summary)` and `interpret(summary, bench_summary, "SPY")` — pure functions in `src/core/market_calc.py`.
6. `rebased({"AAPL": history.points, "SPY": bench_history.points})` (`src/core/market_calc.py`) — puts both series on a shared date range, each starting at 100, so the chart can compare "growth of $100".
7. Returns a `ContextOut` (`src/web_app/api/models.py`).

## 5. `MarketService.get_trend()` — the provider chain

`MarketService.get_trend()` in `src/data/market_client.py`:

1. `self.get_history(symbol, days)` (same file / class):
   - `PersistentCache.get_with_age()` (`src/data/cache.py`, **[EXTERNAL]** `sqlite3`) — a fresh (< 6 hour) cached history returns immediately, `data_status=cached`.
   - Otherwise `self._resolve(symbol, lambda p: p.get_history(symbol, days))` walks the provider list in the configured order (`config.yaml: market.providers`):
     1. **[EXTERNAL]** `AlphaVantageProvider.get_history()` — an HTTP GET to `https://www.alphavantage.co/query` (`httpx` client), paced to 1 request/second and capped at 25/day (`QuotaGuard`, `src/utils/rate_limit.py`). Raises `SymbolNotFoundError`/`ProviderUnavailableError` on failure, which the chain catches and moves on.
     2. **[EXTERNAL]** `YFinanceProvider.get_history()` — the `yfinance` Python package, which itself calls Yahoo Finance's unofficial endpoints. AAPL and SPY are both covered here.
   - The first successful result is cached (`PersistentCache.set()`) and returned as `History(status=live)`.
   - If every provider is down: `PersistentCache.get_stale()` returns the last-good value flagged `stale`; if there is none at all, `MockProvider.get_history()` (same file, pure local computation, no network) returns clearly-labelled sample data.
2. `compute_trend(symbol, points)` (`src/core/market_calc.py`) — a **pure function**: simple moving averages (`sma()`), 1-/3-month `pct_change()`, `annualized_volatility_pct()`, `max_drawdown_pct()`, and a `trend_label()` ("Uptrend"/"Downtrend"/"Sideways / mixed").
3. If the history wasn't live, `trend.notice` is set from `History.__post_init__()` → `build_notice()` (`src/core/freshness.py`), so the caller never has to remember to disclose it.

## 6. The plain-language reading (still no LLM)

`interpret(trend, bench_trend, "SPY")` in `src/core/market_calc.py` builds a list of sentences entirely
from the numbers already computed in step 5 — e.g. "The price is above its 20-day and 50-day averages...",
"Over 3 months it is 10.0 percentage points ahead of SPY (+14.3% vs +4.3%)." — and always ends with "This
describes the past only. Past movement does not predict future results, and it is not a recommendation."
Because every sentence is built with an f-string from already-computed numbers, there is no possibility of
an invented figure — there is no model in this path at all.

## 7. Response back to the browser

`ContextOut` (`src/web_app/api/models.py`): `trend`, `benchmark_trend`, `relative` (percentage-point gaps),
`comparison` (both series rebased to 100), `insights` (the `interpret()` sentences), `data_status`, `notices`.

## 8. Rendering

`MarketAnalysis` (`frontend/src/tabs/Market.jsx`) draws the price chart with 20-/50-day moving-average
overlays (`movingAverage()`, `frontend/src/common.jsx`, applied client-side to the `/trend` points), the
AAPL-vs-SPY rebased comparison chart, the percentage-point "stat chips", and the "What these numbers may
mean" box from `insights`. The segment overview above it (`Overview()`, same file) is a separate call to
`GET /api/market/overview` (`src/web_app/api/routes_market.py:overview`), which runs the identical
`get_trend()` → `compute_trend()` chain for five ETF proxies (SPY/QQQ/IWM/VXUS/BND) in parallel and adds
cross-segment `_takeaways()` (same file).

---

## Call chain at a glance

```
MarketAnalysis (selects AAPL)                       frontend/src/tabs/Market.jsx   — loads quotes, trend and benchmark data for the selected symbol
  → useApi('/api/market/context/AAPL?benchmark=SPY') frontend/src/api.js   — fetches the response and tracks loading/error state for the component
    → apiFetch()                                     frontend/src/http.js   — adds the CSRF header and session cookie to the request
      → fetch()                                      [EXTERNAL] browser   — the browser's network call
        → require_login()                            src/web_app/security.py   — rejects the request unless a valid session cookie is present
        → context()                                  src/web_app/api/routes_market.py   — the /api/market/context route handler
          → enforce()                                 src/web_app/ratelimit.py   — enforces the rate limit for this route
          → MarketService.get_trend('AAPL')            src/data/market_client.py   — fetches the symbol's price history and computes its trend
            → PersistentCache.get_with_age()            src/data/cache.py       [EXTERNAL: sqlite3]   — returns a cached history if it's still fresh
            → AlphaVantageProvider.get_history()         [EXTERNAL: Alpha Vantage HTTP API]   — the primary price-history provider
            → YFinanceProvider.get_history()             [EXTERNAL: yfinance → Yahoo Finance]   — the fallback provider if Alpha Vantage is unavailable
            → compute_trend()                            src/core/market_calc.py   (pure function)   — computes moving averages, returns, volatility and drawdown
          → MarketService.get_trend('SPY')              (same chain, for the benchmark)   — the identical chain above, run again for the benchmark symbol
          → relative_performance() / interpret()        src/core/market_calc.py   (pure functions)   — computes the percentage-point gap and writes the plain-language reading
          → rebased()                                   src/core/market_calc.py   (pure function)   — puts both series on a shared scale starting at 100, for the comparison chart
          → ContextOut(...)                             src/web_app/api/models.py   — shapes the final JSON sent back to the browser
      ← JSON response
  ← chart + "what this may mean" box                    frontend/src/tabs/Market.jsx   — renders the result
```

No agent, no LangGraph node, and no OpenAI/Langfuse/LangSmith call appears anywhere in this chain.
