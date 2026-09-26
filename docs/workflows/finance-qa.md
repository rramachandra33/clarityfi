# Finance Q&A — UI → backend workflow

## Example used

Question: **"What is an ETF?"**

Source test: `tests/unit/test_finance_qa_agent.py::test_relevant_results_produce_a_cited_grounded_answer`
(the same question is also exercised end-to-end, with a real cache hit on the second turn, in
`tests/e2e/test_scenarios.py::test_multi_turn_qa_then_exact_cache_hit_is_disclosed_and_history_survives`).

This is the only tab whose UI shows both a "fresh" and a "cache hit" path, so both are traced below.

---

## 1. User action in the browser

The user types "What is an ETF?" into the chat box (or clicks the built-in suggestion chip with that
text) on the **Finance Q&A** tab and presses Send / Enter.

| Step | Function | File |
|---|---|---|
| Component mounted | `FinanceQA({ userId, onNavigate, isAdmin })` | `frontend/src/App.jsx` |
| Send handler | `ask(question, inputMode)` | `frontend/src/App.jsx` (inside `FinanceQA`) |

`ask()` first checks `navigationTarget(clean)` (`frontend/src/App.jsx`) — a small command parser that lets
a typed phrase like "portfolio analysis" jump to another tab instead of asking the model. For a normal
question this returns `null` and the flow continues.

## 2. Frontend request

```js
apiFetch('/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: userId,
    conversation_id: 'finance-qa',
    question: 'What is an ETF?',
    top_k: 4,
    input_mode: 'text',
    response_mode: 'text',
  }),
})
```

| Function | File | Notes |
|---|---|---|
| `apiFetch(url, options)` | `frontend/src/http.js` | Adds the `X-Clarity-Client: web` header (CSRF defence) and `credentials: 'same-origin'` (sends the session cookie). Dispatches a `clarity:unauthorized` event on a 401. |

The raw `fetch()` call itself is a **[EXTERNAL]** browser Web API, not application code.

## 3. FastAPI request pipeline (runs before the route body)

| Order | Function | File | What it does |
|---|---|---|---|
| 1 | `request_context` middleware | `src/web_app/api/main.py` | Tags the request with an id for logging (`X-Request-ID`). |
| 2 | `CORSMiddleware` | **[EXTERNAL]** Starlette/FastAPI | Allows only the configured origins (`http://127.0.0.1:5173`, `http://localhost:5173`). |
| 3 | `require_login(request)` | `src/web_app/security.py` | App-wide dependency. Checks the `X-Clarity-Client` header on this POST, resolves the session cookie via `AuthService.principal_for()` (`src/web_app/auth_service.py`) → `AuthRepository.user_for_session()` (`src/data/repositories.py`), which hashes the cookie with `token_hash()` (`src/core/auth.py`) and looks it up. |

## 4. Route handler

`chat(request: ChatRequest, http: Request)` in `src/web_app/api/main.py`:

1. `principal_of(http)` (`src/web_app/security.py`) gets the signed-in caller; if `request.user_id` isn't theirs, raises `ApiError(403, ..., "forbidden")`.
2. `enforce(get_services().limiters["chat"], key)` (`src/web_app/ratelimit.py`, keyed by `client_key()`) — a per-user sliding-window rate limit (`SlidingWindowLimiter`, same file).
3. `await workflow.ainvoke(WorkflowRequest(...))` — hands off to the LangGraph workflow (`workflow` is a module-level `ChatWorkflow` built once in `main.py`).
4. Maps the result to a `ChatResponse` (`src/web_app/schemas.py`).

## 5. The LangGraph workflow

`ChatWorkflow.ainvoke()` in `src/workflow/graph.py`:

1. Loads prior turns: `self.memory.history(key)` — `SqliteConversationMemory.history()` (`src/workflow/memory.py`), reading the `messages` table via **[EXTERNAL]** the `sqlite3` standard-library module.
2. Runs the compiled graph: `self.graph.ainvoke({...})`, where `self.graph` is a **[EXTERNAL]** `langgraph.graph.StateGraph` (the `langgraph` package) built in `ChatWorkflow._build()`.

The graph has five nodes, all methods of `ChatWorkflow` in `src/workflow/graph.py`:

| Node | Function | What happens for "What is an ETF?" |
|---|---|---|
| `guard` | `_guard(state)` | `validate_user_input()` (`src/utils/validators.py`) checks length/control characters. `detect_harmful_content()` (`src/core/safety.py`) flags nothing. `looks_like_prompt_injection()` (`src/utils/validators.py`) flags nothing. `redact_text()` (`src/core/pii.py`) — **[EXTERNAL]** Microsoft Presidio (`presidio_analyzer`/`presidio_anonymizer`) + a spaCy NLP model (`en_core_web_lg`) — finds no PII, so `cacheable=True`. |
| `router` | `_route(state)` → `Router.route()` (`src/workflow/router.py`) | `heuristic_route()` matches the `_CONCEPT` pattern ("what is...") and no specialist signal (no ticker, no "my portfolio", no tax/goal/news words), so it returns `intents=["finance_qa"]`, `mode="heuristic"`, `confident=True`. **No LLM call is made** for this question — the rules alone are confident. |
| *(fan-out)* | `_fan_out(state)` | Builds one LangGraph `Send("agent", {...})` for `finance_qa`, with `focus_question()` (`src/workflow/router.py`) returning the question unchanged (only one clause). |
| `agent` | `_agent(payload)` | Looks up `self.agents["finance_qa"]` and calls `agent.arun(request)` with a 45-second timeout (`self.agent_timeout`, from `config.yaml`). |
| `synthesize` | `_synthesize(state)` | Only one agent answered, so this mostly passes its content through, merging its `Data note` block (`split_data_note()`) and ensuring exactly one disclaimer. |
| `output_guard` | `_output_guard(state)` | `redact_output()` (`src/core/pii.py`, Presidio again, high-precision entities only) on the **final merged text**, then `ensure_educational_disclaimer()` (`src/core/disclaimers.py`) guarantees the closing line. |

## 6. The Finance Q&A agent

`FinanceQAAgent.arun()` (inherited from `BaseAgent.arun()`, `src/agents/base.py`) logs `Entering finance_qa agent`, wraps everything in a Langfuse span (`trace_span`, `src/core/tracing.py` — **[EXTERNAL]** the `langfuse` SDK, sending traces to Langfuse's cloud), then calls `FinanceQAAgent._execute()` in `src/agents/finance_qa.py`:

1. **Cache check** — `is_context_dependent("What is an ETF?")` is `False` (it's short but self-contained, an explicit exception), so `use_cache=True`.
   `self.cache.lookup(question, scope, self.embed)` → `QACache.lookup()` (`src/rag/qa_cache.py`).
   - **First turn:** no exact or semantic match yet → returns `None`, proceed to retrieval.
   - **Second, repeated turn** (the e2e test's follow-up): an exact hash match is found (`hashlib.sha256` on the normalized question) and the method returns immediately with `cache_hit="exact"`, `data_status=cached`, and a notice built by `build_qa_cache_notice()` (`src/core/freshness.py`) — **no retrieval and no LLM call happen on a cache hit.**
2. **Retrieval** (first turn only) — `self.kb.search(question, top_k=4, category=None)` → `KnowledgeBase.search()` in `src/rag/knowledge_base.py`:
   - Tries the primary store, `QdrantHybridStore.search()` (`src/rag/qdrant_store.py`) — **[EXTERNAL]** the `qdrant_client` package (embedded local mode or a Qdrant server) for a fused dense+keyword search. Getting the dense vector calls `CachingEmbedder.embed()` (`src/rag/embedding.py`) → `OpenAIClient.embed()` (`src/core/openai_client.py`) — **[EXTERNAL]** an HTTP POST to the OpenAI Embeddings API (`https://api.openai.com/v1/embeddings`).
   - Falls back to `FinanceVectorStore.search()` (`src/rag/vector_store.py`, **[EXTERNAL]** the `faiss` library) only if Qdrant errors or is empty.
   - `prefer_sourced()` (`src/rag/knowledge_base.py`) re-orders near-ties so an article with a real source link wins.
3. **Relevance gate** — results scoring ≥ `score_threshold` (0.25, `config.yaml`) are kept. The ETF article scores well above that, so it passes.
4. **LLM call** — `ainvoke_chat()` (`src/core/llm.py`) is called with `purpose="finance_qa answer"`:
   - Logs `Calling LLM for finance_qa answer (model=gpt-4o-mini)`.
   - Calls `llm.ainvoke(messages, config={"callbacks": [...]})` on a **[EXTERNAL]** `langchain_openai.ChatOpenAI` instance — an HTTP POST to `https://api.openai.com/v1/chat/completions`. The `callbacks` list (`observability_callbacks()`, `src/core/tracing.py`) includes the masked Langfuse and LangSmith tracers — **[EXTERNAL]** `langfuse` and `langsmith` SDKs.
   - Logs `LLM call for finance_qa answer succeeded in N ms`.
   - `to_plain_text()` (`src/core/text.py`) strips markdown; `ensure_educational_disclaimer()` (`src/core/disclaimers.py`) appends the closing line if the model omitted it.
5. **Cache write** (first turn only) — `self.cache.store(...)` → `QACache.store()` (`src/rag/qa_cache.py`) saves the question, its embedding, the answer and the source list, tagged with the current knowledge-base version.
6. Returns an `AgentResult` with `citations` built by `to_citations()` (`src/rag/knowledge_base.py`).

`BaseAgent.arun()` then logs `Leaving finance_qa agent after N ms (...)`.

## 7. Response back to the browser

`main.py:chat()` builds a `ChatResponse` (`src/web_app/schemas.py`) with `answer`, `sources` (from the citations), `cache_hit`, `data_status`, `notice`, `agents_used=["finance_qa"]`, `intents=["finance_qa"]`, `router_mode="heuristic"`.

## 8. Rendering

Back in `FinanceQA` (`frontend/src/App.jsx`), the response is appended to the message list; sources render as a numbered list with publisher/link; if `data.notice` is set (the cache-hit case) it is shown via the amber `DataNotices` component (`frontend/src/common.jsx`) and, in voice mode, spoken aloud by `speakAnswer()`.

---

## Call chain at a glance

```
FinanceQA.ask()                                   frontend/src/App.jsx   — sends the typed question to the backend and adds the reply to the chat log
  → apiFetch('/api/chat')                         frontend/src/http.js   — adds the CSRF header and session cookie to the request
    → fetch()                                     [EXTERNAL] browser   — the browser's network call
      → request_context, CORSMiddleware           src/web_app/api/main.py  [EXTERNAL: Starlette]   — tags the request with an id and checks its origin
        → require_login()                         src/web_app/security.py   — rejects the request unless a valid session cookie is present
          → AuthService.principal_for()           src/web_app/auth_service.py   — resolves the session cookie to the signed-in user
        → chat()                                  src/web_app/api/main.py   — the /api/chat route handler: checks ownership, rate-limits, then runs the workflow
          → enforce() / client_key()               src/web_app/ratelimit.py   — enforces the per-user chat rate limit
          → ChatWorkflow.ainvoke()                 src/workflow/graph.py   [EXTERNAL: langgraph]   — runs the guard → router → agent → synthesize → output-guard graph
            → memory.history()                     src/workflow/memory.py [EXTERNAL: sqlite3]   — loads this conversation's prior turns
            → _guard() → validate_user_input()     src/utils/validators.py   — checks the question's length and strips control characters
                        → detect_harmful_content()  src/core/safety.py   — checks for self-harm or violent content before anything else runs
                        → redact_text()             src/core/pii.py        [EXTERNAL: presidio, spaCy]   — strips personal data from the question before anything else sees it
            → _route() → Router.route()             src/workflow/router.py   — decides which agent(s) should answer
                        → heuristic_route()          src/workflow/router.py   (no LLM needed here)   — pattern-matches the question to an agent without calling a model
            → _agent() → FinanceQAAgent.arun()       src/agents/base.py   — runs the Finance Q&A agent with logging, tracing and a timeout
                        → FinanceQAAgent._execute()  src/agents/finance_qa.py   — the agent's pipeline: cache lookup, retrieval, relevance check, LLM call
                          → QACache.lookup()          src/rag/qa_cache.py   — checks for an existing exact or semantic-match answer to this question
                          → KnowledgeBase.search()    src/rag/knowledge_base.py   — retrieves the most relevant knowledge-base passages
                            → QdrantHybridStore.search()  src/rag/qdrant_store.py [EXTERNAL: qdrant_client]   — runs the dense + keyword hybrid search
                              → CachingEmbedder.embed()   src/rag/embedding.py   — turns the question into a vector, reusing a cached one if available
                                → OpenAIClient.embed()    src/core/openai_client.py [EXTERNAL: OpenAI API]   — calls OpenAI's embeddings endpoint
                          → ainvoke_chat()             src/core/llm.py   — calls the chat model and returns its plain-text reply
                            → ChatOpenAI.ainvoke()      [EXTERNAL: OpenAI API via langchain_openai]   — the actual chat-completion request to OpenAI
                          → QACache.store()            src/rag/qa_cache.py   — saves the new answer so a similar question can be served from cache next time
            → _synthesize()                          src/workflow/graph.py   — merges the agent result(s) into one final answer with one disclaimer
            → _output_guard() → redact_output()       src/core/pii.py       [EXTERNAL: presidio, spaCy]   — strips personal data from the final answer before it leaves the server
          → ChatResponse(...)                        src/web_app/schemas.py   — shapes the final JSON sent back to the browser
      ← JSON response
  ← setMessages(...) / speakAnswer()                 frontend/src/App.jsx   — adds the answer to the chat log and speaks it aloud in voice mode
```
