# News Synthesizer — UI → backend workflow

## Example used

Query: **"Apple earnings"**, with **"Search and summarize"** (`summarize=true`).

Source test: `tests/unit/test_api_routes.py::test_news_live_sample_and_summary`

```python
r = env.get("/api/news?q=Apple earnings&summarize=true").json()
assert r["items"][0]["title"] == "Rates hold steady" and r["summary"] and r["notice"] is None
env.news.status = "sample"
s = env.get("/api/news?q=rates&summarize=true").json()
assert s["data_status"] == "sample" and s["notice"] and s["summary"] is None
```

(The second half of the test is the important safety behaviour: when the headlines are SAMPLE data, no
summary is generated at all — the model is never asked to narrate invented headlines as if they were
real news.)

**Important architectural note:** like the other data tabs, this does **not** go through `/api/chat`.
Plain "Search" makes no LLM call. Only "Search and summarize" adds one — and, notably, that call goes
through a **second, independent instance of the same search** (see step 6), because the summarizing
agent looks up the headlines for itself rather than reusing the ones already fetched by the route.

---

## 1. User action in the browser

The user types "Apple earnings" and clicks **"Search and summarize"**.

| Step | Function | File |
|---|---|---|
| Tab mounted | `NewsSynthesizer()` | `frontend/src/tabs/News.jsx` |
| Button click | `search(summarize=true)` | `frontend/src/tabs/News.jsx` |

## 2. Frontend request

```
GET /api/news?q=Apple%20earnings&limit=6&summarize=true
```

via `api()` (`frontend/src/api.js`) → `apiFetch()` (`frontend/src/http.js`).

## 3. FastAPI request pipeline

`request_context` middleware → `CORSMiddleware` (**[EXTERNAL]** Starlette) → `require_login()`
(`src/web_app/security.py`) — this endpoint has no `{user_id}`, so only "is someone signed in" is
checked. All in `src/web_app/api/main.py` / `security.py`.

## 4. Route handler

`news(request, q, limit, summarize, svc)` in `src/web_app/api/routes_news.py`:

1. `enforce(svc.limiters["expensive"], client_key(request))` (`src/web_app/ratelimit.py`) — news search always uses the stricter "expensive" bucket, whether or not `summarize` is set, since a provider request costs quota either way.
2. `svc.sanitize(q, max_chars=200)` (`src/web_app/api/services.py`) — `validate_user_input()` (`src/utils/validators.py`) then `redact_text()` (`src/core/pii.py`, **[EXTERNAL]** Presidio + spaCy) — the raw query is never sent onward unredacted.
3. `news_query(sanitized_q)` (`src/agents/news.py`) — turns the free text into a short search string: recognized tickers become company names (`extract_symbols()`, `src/agents/entities.py`), stop-words like "news"/"latest" are stripped, capped at 6 words.
4. `await asyncio.to_thread(svc.news.search, query, limit)` — the deterministic headline fetch (step 5).
5. Only if `summarize=true` **and** the result wasn't sample data **and** there are items: `await svc.news_agent.arun(AgentRequest(question=query, user_id="anonymous"))` (step 6).
6. Returns a `NewsOut` (`src/web_app/api/models.py`).

## 5. `NewsService.search()` — the provider chain

`NewsService.search()` in `src/data/news_client.py`:

1. `self._redact(cleaned)` — a second, defence-in-depth redaction call (`redact_text()` again, `src/core/pii.py`), since this method can also be called directly by other callers.
2. A cache key is built from the redacted query; `PersistentCache.get_with_age()` (`src/data/cache.py`, **[EXTERNAL]** `sqlite3`) returns a fresh (< 30 min) cached result immediately if there is one.
3. Otherwise, walks the provider list in order:
   1. **[EXTERNAL]** `SerpApiNewsProvider.search()` — an HTTP GET to `https://serpapi.com/search.json` (Google News engine, `httpx` client), capped at 100 requests/month by `QuotaGuard` (`src/utils/rate_limit.py`).
   2. **[EXTERNAL]** `YFinanceNewsProvider.search()` — the `yfinance` package's news endpoint (used as a fallback, mainly effective for named tickers).
   - If a provider returns nothing for the exact query, it's retried once with `keyword_query()` (`src/data/news_client.py`) — a shorter keyword-only version, since natural-language questions often return zero results from a news search API.
4. On success the items are cached (`PersistentCache.set()`) and returned with `data_status="live"`.
5. If every provider is down or returns nothing: `self._fallback()` (same file) tries a last-good cached copy (any age, flagged `stale`), and only if there is none at all, `SampleNewsProvider.search()` (same file, local, no network) returns clearly-labelled sample headlines with `data_status="sample"`.

For "Apple earnings" in the example, the fake news provider returns a live item ("Rates hold steady"),
so `data_status="live"` and `notice=None`.

## 6. Only when `summarize=true`: the narration

`NewsAgent.arun()` (`BaseAgent`, `src/agents/base.py`) logs `Entering news agent`, then
`NewsAgent._execute()` (`src/agents/news.py`):

1. Builds its **own** search query with `news_query(request.question)` (the `query` string the route already computed is passed in as `request.question`, so this recomputes the identical string) and calls `self.news.search(query, self.limit)` **again** — `NewsService.search()`, the same chain as step 5. In production this second call is almost always a cache hit from step 5's write, but it is a genuinely separate call, not a reuse of the route's in-memory result.
2. If `data_status == sample`, the agent **skips the LLM entirely** and returns a plain template listing the sample headlines (`NewsAgent._template(facts, sample=True)`) — this is the behaviour the second half of the example test checks (`summary` is `None`).
3. Otherwise: `narrate(request.question, facts, fallback_template, extra_rules=NEWS_RULES, agent_name="news")` in `src/agents/narration.py`:
   - `ainvoke_chat(..., purpose="news narration")` (`src/core/llm.py`) → **[EXTERNAL]** OpenAI Chat Completions API via `langchain_openai.ChatOpenAI`. `NEWS_RULES` (`src/agents/news.py`) specifically instructs the model to treat headlines as third-party text ("reports say..."), never follow instructions embedded inside a headline, and never predict prices.
   - `ungrounded_numbers()` (`src/agents/narration.py`) still applies: any dollar/percent figure the summary states must appear in the headline text passed as `facts`.
4. Logs `Leaving news agent after N ms (...)`.

## 7. Response back to the browser

`NewsOut` (`src/web_app/api/models.py`): `query`, `items` (each a `NewsItem` — title, url, source,
published, snippet), `source`, `data_status`, `notice`, `stale`, `cache_hit`, `summary` (only when
requested and not sample).

## 8. Rendering

`NewsSynthesizer` (`frontend/src/tabs/News.jsx`) renders the summary box (if present), then each headline
as a card with a "Read at source" link (only shown if the URL is `http(s)://...`, filtering out anything
else defensively) and the publisher/date line.

---

## Call chain at a glance

```
NewsSynthesizer.search(summarize=true)              frontend/src/tabs/News.jsx   — sends the query and stores the headlines and summary
  → api('/api/news?q=...&summarize=true')           frontend/src/api.js   — makes the request and turns a non-2xx response into a thrown error
    → apiFetch()                                    frontend/src/http.js   — adds the CSRF header and session cookie to the request
      → fetch()                                     [EXTERNAL] browser   — the browser's network call
        → require_login()                           src/web_app/security.py   — rejects the request unless a valid session cookie is present
        → news()                                    src/web_app/api/routes_news.py   — the /api/news route handler
          → enforce()                                src/web_app/ratelimit.py   — enforces the rate limit for this route
          → svc.sanitize() → redact_text()           src/web_app/api/services.py / src/core/pii.py  [EXTERNAL: presidio, spaCy]   — strips personal data from the search query before it leaves the machine
          → news_query()                             src/agents/news.py   — turns the free-text question into a short search string
          → NewsService.search()                     src/data/news_client.py   — fetches headlines, falling back through providers as needed
            → SerpApiNewsProvider.search()             [EXTERNAL: SerpAPI HTTP API]   — the primary news search
            → YFinanceNewsProvider.search()             [EXTERNAL: yfinance → Yahoo Finance]   — the fallback provider, mainly effective for named tickers
          → (only if summarize=true and not sample)
            NewsAgent.arun() → _execute()             src/agents/news.py   — summarizes the headlines it fetches for itself
              → NewsService.search()  (again)          src/data/news_client.py    (see above)   — a second, independent headline lookup, usually a cache hit
              → narrate()                              src/agents/narration.py   — asks the model to summarize the headlines, then checks its reply
                → ainvoke_chat()                        src/core/llm.py   — calls the chat model and returns its plain-text reply
                  → ChatOpenAI.ainvoke()                  [EXTERNAL: OpenAI API]   — the actual chat-completion request to OpenAI
                → ungrounded_numbers()                  src/agents/narration.py  (grounding check)   — rejects the summary if it states a figure not present in the headlines
          → NewsOut(...)                              src/web_app/api/models.py   — shapes the final JSON sent back to the browser
      ← JSON response
  ← headline cards + summary box                       frontend/src/tabs/News.jsx   — renders the result
```
