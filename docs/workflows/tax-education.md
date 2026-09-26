# Tax Education — UI → backend workflow

## Example used

Question: **"What is a Roth IRA?"**

Source test: `tests/unit/test_tax_agent.py::test_reads_the_tax_slice_of_the_knowledge_base_with_a_strict_prompt`

```python
a = tax_agent()
r = ask(a, "What is a Roth IRA?")
assert r.ok and r.agent is AgentName.tax and a.store.calls[0]["category"] == "tax"
assert a.llm.messages[0][0].content == TAX_SYSTEM_PROMPT and "Never calculate a person's tax bill" in TAX_SYSTEM_PROMPT
assert r.citations[0].source == "Investopedia" and r.citations[0].url == "https://ex.com/roth"
```

**Important architectural note:** this tab is built on the **same `/api/chat` → LangGraph workflow →
router** pipeline as Finance Q&A (see `finance-qa.md` for the shared parts, which aren't repeated in full
here). The only differences are (1) the router sends this specific question to the **tax** agent instead
of `finance_qa`, and (2) `TaxEducationAgent` is `FinanceQAAgent` with a stricter prompt, a narrower
knowledge-base slice, and an always-appended caveat.

---

## 1. User action in the browser

The user types "What is a Roth IRA?" (or clicks one of the example-question chips) on the **Tax
Education** tab and presses Send.

| Step | Function | File |
|---|---|---|
| Tab mounted | `TaxEducation({ userId })` | `frontend/src/tabs/Tax.jsx` |
| Send handler | `ask(text)` | `frontend/src/tabs/Tax.jsx` |

## 2. Frontend request

```js
api('/api/chat', {
  method: 'POST',
  body: { user_id: userId, conversation_id: 'tax-education', question: 'What is a Roth IRA?', top_k: 4 },
})
```

Note this uses a fixed `conversation_id: 'tax-education'` — separate history from the Finance Q&A tab's
`'finance-qa'` conversation, even though both call the exact same `/api/chat` endpoint. `api()`
(`frontend/src/api.js`) wraps `apiFetch()` (`frontend/src/http.js`), same as every other POST.

## 3–4. FastAPI pipeline and route handler

Identical to Finance Q&A's steps 3–4: `request_context` → `CORSMiddleware` (**[EXTERNAL]** Starlette) →
`require_login()` (`src/web_app/security.py`) → `chat()` in `src/web_app/api/main.py` → ownership check
via `principal_of()` → `enforce()` (`src/web_app/ratelimit.py`) → `ChatWorkflow.ainvoke()`
(`src/workflow/graph.py`). See `finance-qa.md` for the full detail of this shared part.

## 5. Where this diverges: the router

`Router.route()` → `heuristic_route()` in `src/workflow/router.py`. The question is checked against
each specialist pattern; `_TAX` (a regex including `roth`, `traditional ira`, `401k`, `hsa`, `capital
gains`, etc.) matches "roth", so `signals["tax"] = ["roth"]`. No other pattern matches (no personal-
portfolio words, no ticker, no goal or news language), so `intents = ["tax"]`, `mode="heuristic"`,
`confident=True` — **no LLM call is needed to route this question.**

`ChatWorkflow._plan()` (`src/workflow/graph.py`) then maps intent `"tax"` straight to the **agents["tax"]**
entry (`TaxEducationAgent`, registered in `src/web_app/api/main.py`) — there's a dedicated tax agent
registered, so the "fall back to finance_qa over the tax category" branch in `_plan()` is not used here;
that branch only fires if no `tax` agent were registered at all.

## 6. The Tax Education agent

`TaxEducationAgent.arun()` (`BaseAgent`, `src/agents/base.py`) logs `Entering tax agent`, then
`TaxEducationAgent._execute()` in `src/agents/tax.py`:

1. Calls `super()._execute(request)` — i.e. `FinanceQAAgent._execute()` in `src/agents/finance_qa.py`,
   the **exact same pipeline documented in `finance-qa.md` step 6** (cache lookup, hybrid retrieval,
   relevance gate, LLM call, cache write), with two class-level overrides that change its behaviour:
   - `system_prompt = TAX_SYSTEM_PROMPT` (`src/agents/tax.py`) — a stricter prompt: never compute a
     person's tax, never state a dollar limit/rate not present in the retrieved passages, say that rules
     change and vary by state/year.
   - `default_category = "tax"` — since the route didn't specify a category, `FinanceQAAgent._execute()`
     uses this as the category, so `self.kb.search(question, top_k, category="tax")` →
     `KnowledgeBase.search()` (`src/rag/knowledge_base.py`) → `resolve_categories("tax")`
     (`src/rag/categories.py`) expands the group to `("taxes_and_investing", "tax_efficiency",
     "tax_education", "retirement")` before it reaches Qdrant/FAISS — the same external retrieval and
     LLM calls as Finance Q&A (**[EXTERNAL]** OpenAI embeddings + chat completions, Qdrant/FAISS), just
     scoped to that slice of the knowledge base.
2. Back in `TaxEducationAgent._execute()`: `is_personal_tax_question("What is a Roth IRA?")`
   (`src/agents/tax.py`, a regex for things like "how much tax will I owe") returns `False` — this is a
   conceptual question, not a request to compute someone's tax.
3. Appends `TAX_NOTICE` (`src/agents/tax.py`) — "Tax rules change and depend on your situation..." — to
   the answer, after the (possibly cached) content, unless it's already present (so a cached answer
   never gets the caveat glued on twice).
4. Logs `Leaving tax agent after N ms (...)`.

## 7. Synthesis and output guard

Same as Finance Q&A: `ChatWorkflow._synthesize()` and `_output_guard()` (`src/workflow/graph.py`) merge
the single agent's answer, run `redact_output()` (`src/core/pii.py`, **[EXTERNAL]** Presidio) on the final
text, and guarantee the closing disclaimer via `ensure_educational_disclaimer()`
(`src/core/disclaimers.py`).

## 8. Response back to the browser and rendering

`main.py:chat()` returns a `ChatResponse` (`src/web_app/schemas.py`) with `agents_used=["tax"]`,
`intents=["tax"]`, `router_mode="heuristic"`, `sources` (from `Citation` objects built by `to_citations()`,
`src/rag/knowledge_base.py` — here the Investopedia Roth IRA article). `TaxEducation`
(`frontend/src/tabs/Tax.jsx`) renders the answer text, the sources list (as links when a URL is present),
and any `notice` through the shared `DataNotices` component (`frontend/src/common.jsx`).

---

## Call chain at a glance

```
TaxEducation.ask()                                  frontend/src/tabs/Tax.jsx   — sends the typed question to the backend and stores the reply
  → api('/api/chat')                                frontend/src/api.js   — makes the request and turns a non-2xx response into a thrown error
    → apiFetch()                                    frontend/src/http.js   — adds the CSRF header and session cookie to the request
      → fetch()                                     [EXTERNAL] browser   — the browser's network call
        → require_login()                           src/web_app/security.py   — rejects the request unless a valid session cookie is present
        → chat()                                    src/web_app/api/main.py   — the shared /api/chat route handler (the same one Finance Q&A uses)
          → enforce()                                src/web_app/ratelimit.py   — enforces the per-user chat rate limit
          → ChatWorkflow.ainvoke()                    src/workflow/graph.py   [EXTERNAL: langgraph]   — runs the guard → router → agent → synthesize → output-guard graph
            → _guard() → redact_text()                src/core/pii.py         [EXTERNAL: presidio, spaCy]   — strips personal data from the question before anything else sees it
            → _route() → heuristic_route()            src/workflow/router.py     ("tax" matched; no LLM needed)   — matches the question to the tax agent by keyword
            → _agent() → TaxEducationAgent.arun()      src/agents/base.py   — runs the Tax Education agent with logging, tracing and a timeout
                        → TaxEducationAgent._execute() src/agents/tax.py   — adds the tax-specific prompt, category and caveat around the shared Q&A pipeline
                          → FinanceQAAgent._execute()  src/agents/finance_qa.py   (category="tax")   — the shared cache/retrieval/LLM pipeline, scoped to tax content
                            → QACache.lookup()          src/rag/qa_cache.py   — checks for an existing cached answer to this question
                            → KnowledgeBase.search()    src/rag/knowledge_base.py   — retrieves the most relevant tax-category passages
                              → resolve_categories()      src/rag/categories.py   — expands "tax" into its underlying knowledge-base categories
                              → QdrantHybridStore.search()  src/rag/qdrant_store.py [EXTERNAL: qdrant_client]   — runs the dense + keyword hybrid search
                                → CachingEmbedder.embed()   src/rag/embedding.py   — turns the question into a vector
                                  → OpenAIClient.embed()      [EXTERNAL: OpenAI Embeddings API]   — calls OpenAI's embeddings endpoint
                            → ainvoke_chat()             src/core/llm.py   — calls the chat model and returns its plain-text reply
                              → ChatOpenAI.ainvoke()       [EXTERNAL: OpenAI Chat Completions API]   — the actual chat-completion request to OpenAI
                            → QACache.store()            src/rag/qa_cache.py   — saves the new answer for next time
                          → is_personal_tax_question()  src/agents/tax.py   — detects whether the user is asking about their own tax situation
                          → (append TAX_NOTICE)          src/agents/tax.py   — adds the "rules change and depend on your situation" caveat to the answer
            → _synthesize()                            src/workflow/graph.py   — merges the agent result into the final answer
            → _output_guard() → redact_output()         src/core/pii.py         [EXTERNAL: presidio, spaCy]   — strips personal data from the final answer before it leaves the server
          → ChatResponse(...)                          src/web_app/schemas.py   — shapes the final JSON sent back to the browser
      ← JSON response
  ← render answer + sources + notice                   frontend/src/tabs/Tax.jsx   — renders the result
```
