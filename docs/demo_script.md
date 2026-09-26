# Demo video script (5–10 minutes)

Follows the spec in the "demo-video" doc, in order. Two small corrections from that doc: the demo user id is
**`learner-101`** (the doc says "starter-101"; the app's ten synthetic accounts are `learner-101`…`learner-110`,
no `starter-101` exists), and the server log is **`/tmp/clarity-api.log`** (the doc says "clarity-fi.log").

`learner-101` is "Alex Rivera" — holds QQQ/VTI/NVDA/TSLA + cash, risk comfort currently set to Conservative, and
already has two saved goals ("Home down payment" and "Car payment"). Every step below was run live against the
real app while writing this script, so the numbers, routing decisions, and log lines are real, not illustrative.

## Setup (before you hit record)

1. Make sure nothing else is on port 8000/5173 (`lsof -nP -iTCP:8000 -sTCP:LISTEN`, `-iTCP:5173`).
2. `./start.command` — wait for both the API and the Vite dev server to come up.
3. Open a terminal to `tail -f /tmp/clarity-api.log` so you can show real lines right after each action.
4. Open a browser tab to `https://us.cloud.langfuse.com` → **Raju's Organization → My Project → Tracing** — the
   trace list updates live as you use the app, so you can open the newest one right after each test.
5. Optional: `docs/design.md`'s System architecture diagram has a static view of the LangGraph if you want a
   second image alongside the live Langfuse graph (see step 9).

## Shot list

| # | Spec item | Do | Say | Real evidence (captured live) |
|---|---|---|---|---|
| 1 | Start from the login screen | Load the app fresh (sign out first if a session is active) | "ClarityFI is a multi-agent finance education assistant. Real accounts — hashed passwords, per-user data isolation — or a demo id to explore." | — |
| 2 | User `learner-101` as the demo user id | Click the `learner-101` demo chip (password shown on screen) and sign in | "Ten synthetic demo users ship with the app; I'm using `learner-101`." | `POST /api/auth/login HTTP/1.1" 200 OK` |
| 3 | Finance QA, sample question **not in cache** | Ask "What is a Roth IRA?" | "First time this exact question has been asked. Watch the router pick an agent, then a real model call." | `Routing decision: intents=['tax'] mode=heuristic ... reason=signals: tax=roth` → `Entering tax agent` → `Calling LLM for tax answer (model=gpt-4o-mini)` → `Leaving tax agent after 5360 ms (error=None, cache=None)`. *(Router picked the Tax agent specifically, not generic Q&A — a good detail to call out: the router recognized "Roth" as a tax-account signal.)* |
| 4 | Example where data is read from **cache** | Ask the *exact same* question again | "Same question, served from the exact-match cache — no model call this time." | `Entering tax agent` → `Leaving tax agent after 5 ms (error=None, cache=exact)` — no `Calling LLM` line at all. UI shows an amber notice: *"Answered from a saved response to this same question (about 1 minute old). It was not regenerated."* |
| 5 | Typing "portfolio" in Finance QA takes you to the right tab | Type exactly `portfolio` into the Finance Q&A box and submit | "The box doubles as a command bar — this never touches the backend." | Confirmed live: **no** `/api/chat` line appears in the log for this action — only the Portfolio tab's own `GET /api/portfolio/learner-101/analysis` load. |
| 6 | Showcase each tab for this user | Tour Portfolio → Market → Goals → News → Tax → Glossary (steps 10–13 below have their own real captures) | — | — |
| 7 | Multi-turn conversations with different agents | Ask "What is AAPL trading at, and what is an ETF?" | "One question, two agents, running in parallel." | `Routing decision: intents=['market', 'finance_qa'] ... reason=signals: market=symbols:AAPL, finance_qa=concept clause` then **both** `Entering market agent` and `Entering finance_qa agent` logged in the same second — each calls the model independently, both `Leaving ... agent` within ~3.7s of each other. Reply merges into "Market" + "Background" sections in one bubble. |
| 8 | Langfuse screenshot | Switch to Langfuse → open the trace for step 7 → **Tree** view | "Every turn is one trace. Here's `workflow.chat` → `workflow.route` → `agent.market` and `agent.finance_qa` as sibling spans, each with its own nested `ChatOpenAI` generation, cost and token count." | Real trace `e2ae9b9fb9b0cda15bea9d56cb76e3f9`: latency 3.78s, cost $0.000339, 1,783 tokens. *(Bonus talking point: the trace's own preview panel shows the number in "the change percentage is ⟨PERSON⟩" — Langfuse's own masking is deliberately conservative for stored traces, separate from and stricter than what the user actually sees in the UI, which correctly showed "-0.64%".)* |
| 9 | LangGraph image | Same trace → switch to **Graph** view → **Expanded** | "This is the literal LangGraph for this run, not a hand-drawn diagram." | Langfuse renders the actual node graph from the run's telemetry: `__start__` → `workflow.route` → two parallel branches (`agent.market` and `agent.finance_qa`, each with its `ChatOpenAI` child) → `__end__`. |
| 10 | Portfolio analysis demonstration | Portfolio tab: donut, holdings table (QQQ/VTI/TSLA/NVDA + cash, $48,934 total), diversification score (70/100, "Growth risk"), risk-comfort dropdown | "Every number here is code-computed, not model-generated." | `GET /api/portfolio/learner-101/analysis` 200 (also triggers `/comparison?benchmark=SPY`). Live bonus: Alpha Vantage's quota was exhausted mid-test — you'll see `alphavantage unavailable: Alpha Vantage quota or rate limit reached` in the log, followed by a clean yFinance fallback with no user-facing error. Good moment to point at the provider-fallback design. |
| 11 | Market data integration | Market tab: segment overview bars, 1M/3M returns | "Alpha Vantage first, yFinance fallback, then last-known — always labelled." | Same AV-quota fallback as above; the UI shows an amber **"SAVED COPY"** badge on every segment row plus the banner *"SPY, QQQ, IWM, VXUS and BND are a saved copy rather than a fresh lookup."* — a live, unstaged example of the transparency rule. |
| 12 | Goal planning example | Goals tab: point at the two existing saved goals, then *Project and explain* on "Home down payment" | "Real saved goals from the seed data, plus a fresh three-scenario projection with a narrated explanation." | `Entering goal_planning agent` → `Calling LLM for goal_planning narration` → `Leaving goal_planning agent after 4642 ms`. UI result: "Short of target — Projected $78,000 vs. target $80,000." |
| 13 | Glossary example | Glossary tab: search a term | "203 terms, read-only reference, loaded once — no agent call, it's a static lookup." | No backend agent line for this one; that's the point. |

## MCP server: tested live with the actual Claude Code CLI

The four read-only MCP tools (`get_quote`, `analyze_portfolio`, `search_knowledge_base`, `project_goal`,
`docs/mcp.md`) aren't just unit-tested — they were exercised through a real, independent Claude client while
writing this script: `claude mcp add clarityfi -s local -- .venv/bin/python -m src.mcp_server.server`, then a
fresh `claude -p "..."` process for each tool. That's a second, separate Claude process talking to the running
app purely over MCP/stdio, with no shared code path to the browser demo above.

| # | Do (on camera) | Real result (captured live) |
|---|---|---|
| a | `claude mcp add clarityfi -s local -- .venv/bin/python -m src.mcp_server.server`, then `claude mcp get clarityfi` | `Status: ✔ Connected` |
| b | `claude -p "Use the clarityfi MCP get_quote tool to get the current price of AAPL and VTI."` | Returned a table with price, change %, source (`yfinance`), and `data_status: cached` with an explicit "about 6 minutes old" notice for both symbols — proves freshness disclosure works even over MCP, not just the web UI. |
| c | `claude -p "Use the clarityfi MCP analyze_portfolio tool for user learner-101."` | Total value $48,953.42, diversification score 69.6, risk label "Growth" — matches the Portfolio tab from step 10 above (small differences are just quote timing). |
| d | `claude -p "Use the clarityfi MCP search_knowledge_base tool to search for 'Roth vs traditional IRA', top_k 2."` | Returned "Traditional IRA" and "Roth IRA", both sourced to Investopedia. |
| e | `claude -p "Use the clarityfi MCP project_goal tool: target_amount 50000, years 8, current_savings 5000, monthly_contribution 300."` | Three scenarios ("Below your assumption" / "Your assumption" / "Above your assumption"); the middle one reported not on track. |

Say while doing this: "The same read-only tools the web app uses are also exposed over MCP, so any MCP client —
Claude Code, Claude Desktop, an IDE agent — can call them directly, with the same disclaimers and freshness
notices. This is a completely separate client from the browser session above." If you'd rather demo it inside
the **Claude Desktop app UI** instead of the CLI, `docs/mcp.md` has the config block and the same four example
prompts — that one needs a one-time edit to `claude_desktop_config.json` (or run `python3
scripts/install_claude_desktop.py`) and a restart of Claude Desktop, which I can't drive from here since it's a
separate GUI app.

Cleanup after recording, if you don't want the server left registered: `claude mcp remove clarityfi -s local`.

## Wrap-up (optional, if time allows)

- News tab: search "interest rates" → *Search and summarize* (untrusted third-party headlines, sourced links).
- Tax tab: "How does a Roth IRA work?" then a personal-tax question (concept explained; personal computation
  politely refused).
- HD voice toggle in Finance Q&A: ask a question by voice (Whisper transcription, OpenAI TTS playback).

## Numbers to quote

6 agents · 230 knowledge-base articles · 1,276 backend + 42 frontend tests · 99% coverage · cache hit ≈ 5 ms vs
≈ 5.4 s fresh (measured live in step 3/4 above) · retrieval 10/10 on the benchmark set.
