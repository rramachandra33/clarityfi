# Portfolio Analysis — UI → backend workflow

## Example used

User: **`learner-105`** (holds VTI, VXUS and BND — a three-fund portfolio; see `src/data/seed.py`), opening the Portfolio Analysis tab, which loads the saved analysis, then discloses when a price stops being live.

Source test: `tests/unit/test_api_routes.py::test_analysis_is_deterministic_and_discloses_non_live_prices`

```python
ok = env.get("/api/portfolio/learner-105/analysis").json()
assert ok["metrics"]["total_value"] > 0 and ok["data_status"] == "live" and ok["explanation"] is None
env.market.quotes["VTI"] = quote("VTI", 300.0, status=DataStatus.stale)
stale = env.get("/api/portfolio/learner-105/analysis").json()
assert stale["data_status"] == "stale" and stale["notice"] and stale["notices"]
```

**Important architectural note:** this tab does **not** go through `/api/chat` or the LangGraph
router at all. It calls the portfolio REST endpoints directly, and by default runs with **no LLM
call whatsoever** — every number is computed in Python. An LLM is only invoked if the user clicks
**"Explain in plain language"** (`explain=true`), which is covered as a second example below.

---

## 1. User action in the browser

The user clicks the **Portfolio Analysis** tab.

| Step | Function | File |
|---|---|---|
| Tab mounted | `PortfolioAnalysis({ userId })` | `frontend/src/tabs/Portfolio.jsx` |
| Data fetch (auto, on mount) | `useApi(`/api/portfolio/${id}/analysis`)` | `frontend/src/api.js` (used from `Portfolio.jsx`) |

`useApi()` (`frontend/src/api.js`) calls `api(path)`, which calls `apiFetch()` (`frontend/src/http.js`).
The Portfolio tab also fires two more `useApi()` calls in parallel: `/api/users/${id}` (the profile, for the
risk dropdown) and, once the analysis is non-empty, `/api/portfolio/${id}/comparison?benchmark=SPY` (the
"how today's mix would have done" panel — see the note at the end of this file).

## 2. Frontend request

```
GET /api/portfolio/learner-105/analysis
```

(`?explain=true` is appended if the user has clicked "Explain in plain language".)

## 3. FastAPI request pipeline

Same as every route: `request_context` middleware → `CORSMiddleware` (**[EXTERNAL]** Starlette) →
`require_login()` (`src/web_app/security.py`), which also enforces that `learner-105` in the path
belongs to the signed-in caller. All in `src/web_app/api/main.py` / `src/web_app/security.py`.

## 4. Route handler

`analysis(user_id, request, explain, svc)` in `src/web_app/api/routes_portfolio.py`:

1. `enforce(svc.limiters["expensive" if explain else "read"], client_key(request))` (`src/web_app/ratelimit.py`) — a cheaper rate-limit bucket for the plain read, a stricter one when `explain=true` (it costs an LLM call).
2. `svc.require_user(user_id)` (`src/web_app/api/services.py`) — 404s if the id doesn't exist.
3. `await svc.portfolio_agent.analyze(profile.user_id)` — the deterministic engine, always run.
4. Only if `explain=true`: `await svc.portfolio_agent.arun(AgentRequest(question=EXPLAIN_QUESTION, ...))`.
5. `_out(...)` (same file) shapes the response into `PortfolioAnalysisOut` (`src/web_app/api/models.py`).

## 5. The deterministic engine (no LLM)

`PortfolioAgent.analyze()` in `src/agents/portfolio.py`:

1. `self._load(request)` — reads the saved portfolio: `PortfolioRepository.get()` (`src/data/repositories.py`, **[EXTERNAL]** `sqlite3`) and the user's stated risk tolerance via `UserRepository.get()` (same file).
2. `self.market.get_quotes(["VTI", "VXUS", "BND"])` → `MarketService.get_quotes()` in `src/data/market_client.py`. For each ticker this calls `MarketService.get_quote()`, which walks the provider chain (see `market-analysis.md` for the full chain): a fresh TTL cache hit, else **[EXTERNAL]** Alpha Vantage (`AlphaVantageProvider.get_quote()`, HTTP to `alphavantage.co`), else **[EXTERNAL]** yFinance (`YFinanceProvider.get_quote()`, the `yfinance` package → Yahoo Finance), else the last-good cached value (flagged `stale`), else sample data (never used for valuation, see step 3).
3. Any quote with `data_status == sample` is excluded from `real` — the code explicitly never uses invented prices to value a portfolio; that holding falls back to its cost basis instead, and a notice is added.
4. `compute_portfolio_metrics(portfolio, prices, stated_risk)` — a **pure function** in `src/core/portfolio_calc.py`. Computes: total/invested/cash value, allocation by asset class and by holding, weighted expense ratio and annual fee estimate, unrealized gain/loss, a diversification score (0–100, effective-position count with a broad-fund look-through), a risk score/label, and plain-language notes (e.g. "the mix looks more aggressive than your stated moderate comfort level").
5. `self._rows(...)` (same file) builds the per-holding table shown in the UI, largest first, each row's `price_source` set by `provider_label()` (`src/core/freshness.py`).
6. Returns a `PortfolioAnalysis` dataclass — no `AgentResult`, no LLM, nothing narrated.

**On the second call** in the example (`VTI` now `stale`), step 2 returns a `Quote` whose `data_status` is `stale`; step 3's `notices` list gains an entry, `combine_status()` (`src/core/freshness.py`) sets the overall `status` to the worse of the two, and the route returns `data_status: "stale"` with a non-empty `notice`/`notices` — nothing else in the pipeline changes.

## 6. Only when `explain=true`: the narrated version

`PortfolioAgent.arun()` (from `BaseAgent`, `src/agents/base.py`) logs `Entering portfolio agent`, then `PortfolioAgent._execute()` (`src/agents/portfolio.py`):

1. Calls `self.analyze()` again (the same deterministic path above).
2. Builds a `facts` dict from the metrics (total value, allocation, top holdings, scores, notes — plain numbers only, no free text the model could misquote).
3. `narrate(request.question, facts, fallback_template, agent_name="portfolio")` in `src/agents/narration.py`:
   - `ainvoke_chat(..., purpose="portfolio narration")` (`src/core/llm.py`) → **[EXTERNAL]** OpenAI Chat Completions API via `langchain_openai.ChatOpenAI`.
   - `ungrounded_numbers(text, facts)` (`src/agents/narration.py`) scans the model's reply for any dollar/percent figure that is **not** present in `facts`; if it finds one, the LLM's text is discarded and `PortfolioAgent._template(facts)` (a deterministic string) is used instead. This is the safety net that stops the model from inventing a number.
4. Logs `Leaving portfolio agent after N ms (...)`.

## 7. Response back to the browser

`PortfolioAnalysisOut` (`src/web_app/api/models.py`): `metrics`, `holdings`, `data_status`, `notice`/`notices`, `priced_with`/`unpriced`, `stated_risk_tolerance`, and `explanation` (only when `explain=true`).

## 8. Rendering

`PortfolioAnalysis` (`frontend/src/tabs/Portfolio.jsx`) draws the allocation donut (`donutBackground()`), the per-holding list, the diversification score card, and — if `explanation` is present — the "In plain language" box. Any `notices` render through the shared `DataNotices` component (`frontend/src/common.jsx`), and a non-live `data_status` shows the `StatusBadge` (same file).

---

## Call chain at a glance

```
PortfolioAnalysis (mount)                          frontend/src/tabs/Portfolio.jsx   — loads the saved portfolio's analysis when the tab opens
  → useApi('/api/portfolio/{id}/analysis')          frontend/src/api.js   — fetches the response and tracks loading/error state for the component
    → apiFetch()                                    frontend/src/http.js   — adds the CSRF header and session cookie to the request
      → fetch()                                     [EXTERNAL] browser   — the browser's network call
        → require_login()                           src/web_app/security.py   — rejects the request unless a valid session cookie is present
        → analysis()                                src/web_app/api/routes_portfolio.py   — the /api/portfolio/{user}/analysis route handler
          → enforce()                                src/web_app/ratelimit.py   — enforces the rate limit for this route
          → PortfolioAgent.analyze()                 src/agents/portfolio.py   — prices the holdings and computes every metric, with no LLM involved
            → PortfolioRepository.get()               src/data/repositories.py  [EXTERNAL: sqlite3]   — reads the saved holdings and cash
            → MarketService.get_quotes()              src/data/market_client.py   — fetches a current price for each holding
              → AlphaVantageProvider.get_quote()        [EXTERNAL: Alpha Vantage HTTP API]   — the primary quote provider
              → YFinanceProvider.get_quote()             [EXTERNAL: yfinance → Yahoo Finance]   — the fallback provider if Alpha Vantage is unavailable
            → compute_portfolio_metrics()             src/core/portfolio_calc.py   (pure function)   — computes value, allocation, fees, diversification and risk scores
          → (only if explain=true)
            PortfolioAgent.arun() → _execute()        src/agents/portfolio.py   — narrates the already-computed metrics
              → narrate()                             src/agents/narration.py   — asks the model to explain the numbers, then checks its reply for invented figures
                → ainvoke_chat()                       src/core/llm.py   — calls the chat model and returns its plain-text reply
                  → ChatOpenAI.ainvoke()                 [EXTERNAL: OpenAI API]   — the actual chat-completion request to OpenAI
                → ungrounded_numbers()                 src/agents/narration.py  (grounding check)   — rejects the narration if it states a figure not in the metrics
          → PortfolioAnalysisOut(...)                 src/web_app/api/models.py   — shapes the final JSON sent back to the browser
      ← JSON response
  ← render donut / holdings / score card              frontend/src/tabs/Portfolio.jsx   — renders the result
```

## Note: the benchmark comparison panel

The same tab also shows "How today's mix would have done vs a benchmark", fetched from
`GET /api/portfolio/{user}/comparison` (`src/web_app/api/routes_portfolio.py:comparison`), which replays
the saved holdings over past prices via `value_series()`/`interpret()` (`src/core/portfolio_history.py`) and
`compute_trend()`/`rebased()`/`relative_performance()` (`src/core/market_calc.py`). This is also fully
deterministic (no LLM) and is documented in the same style in `market-analysis.md`, since it shares the
market-data chain.
