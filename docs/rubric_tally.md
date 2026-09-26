# Rubric tally (self-assessment)

Source: `Grading_Rubric_AI_Finance_Assistant.xlsx` (Excellent = 100%, Good 75%, Satisfactory 50%, Needs improvement 25%). Scores below are my honest estimate of where the evidence lands, not a claim of the grade. The rubric names the UI "Streamlit"; the problem statement's architecture (and the provided reference code) use React, so the web interface criterion is assessed on the React app.

## Technical Implementation (40)

| Criterion (weight) | Evidence | Est. | Gap / risk |
|---|---|---|---|
| Multi-agent architecture (10) | 6 distinct agents on a shared `BaseAgent`; router splits multi-part questions so each agent answers its own clause; parallel fan-out with per-agent timeouts and contained failures; synthesizer merges citations, notices and disclaimers; tax agent specialises Q&A | 10 | "Agent-to-agent" communication is via the shared graph state and synthesizer, not free-form agent chat |
| LangGraph workflow (10) | `StateGraph` guard → route → parallel agents (`Send`) → synthesize → output guard; state carries intents, results, notices; fallbacks on every stage; router accuracy measured (88% on a fresh set) and regression-tested | 10 | No LangGraph checkpointer (own SQLite memory by design) |
| RAG implementation (8) | 230 articles; Qdrant dense + sparse with RRF, FAISS fallback, category filter, relevance gate, exact + semantic answer cache with guards; live retrieval 10/10 top-3 | 8 | All 230 articles now carry a real, verified source (was 90 placeholder-sourced) |
| Real-time data integration (7) | Alpha Vantage → yFinance → last-good → sample with pacing, quota tracking, TTL cache; SerpAPI news with same fallbacks; every non-live value labelled | 7 | Free-tier quota (25/day) means yFinance often serves |
| MCP server (5) | `src/mcp_server`: 4 read-only tools (`get_quote`, `analyze_portfolio`, `search_knowledge_base`, `project_goal`) over stdio with annotations, input validation, PII-safe search, data-provenance notices; 22 tests through a real in-memory MCP client + live stdio run; Claude Desktop config in `docs/mcp.md`; confirmed connected and working from inside Claude Desktop itself (2026-09-26) | 5 | Read-only by design |

## User Experience & Interface (25)

| Criterion (weight) | Evidence | Est. | Gap / risk |
|---|---|---|---|
| Web interface (10) | 7-tab React app with navigation, responsive layout, user gate, error/loading states, data-status badges; checked in a real browser | 10 | Only lightly tested on small screens |
| Conversational flow (8) | Persistent history per conversation (survives restart), follow-ups understood, sources on every answer, voice in/out, cache and data notices in-line | 8 | Context is last several turns, not a summary of long chats |
| Data visualization (7) | Interactive charts (hover/keyboard crosshair, per-series tooltips, dated axes, legends, low/high): price with 20/50-day averages, symbol vs benchmark rebased to 100, goal scenarios against a dashed target line, allocation donut with hover-highlighted class bars, market-segment return bars; every non-live value badged | 7 | No zoom/pan or export; charts are hand-built SVG, not a charting library |

## Financial Domain Knowledge (20)

| Criterion (weight) | Evidence | Est. | Gap / risk |
|---|---|---|---|
| Educational content (8) | 230 curated articles across ~50 categories, 200+ term glossary, category filter, citations | 8 | All 230 articles carry a real, verified source as of 2026-09-25 (was 90 placeholder-sourced); content is otherwise as supplied |
| Portfolio analysis (7) | Value, allocation (holding/class), gain/loss, weighted expense ratio + fee, diversification with broad-fund look-through, risk score/label, stated-risk mismatch, what-if, per-holding price provenance, and a hypothetical 'today's mix vs a benchmark' replay (rebased chart, gaps, drawdown, volatility, coverage, computed reading) | 7 | Replay uses today's holdings (not actual trade history) |
| Market intelligence (5) | Segment overview (US large/tech/small, international, bonds) with takeaways; benchmark comparison (percentage-point gaps, rebased chart); computed plain-language interpretation (trend vs averages, distance from high, volatility band, relative performance); quotes, SMA 20/50, returns, drawdown, index-proxy explanations | 5 | Interpretation is rule-based and descriptive by design (no forecasts, no news/fundamentals linkage) |

## Code Quality & Documentation (15)

| Criterion (weight) | Evidence | Est. | Gap / risk |
|---|---|---|---|
| Code organization (5) | `src/{agents,core,data,rag,workflow,web_app,utils,mcp_server}`; single responsibility; typed models; errors with stable codes; config in YAML + `.env`; secure-by-default routing (login, ownership and CSRF enforced once, application-wide, with a test that fails if any new route is left open) | 5 | Legacy `openai_client.py` still used for embeddings; `main.py` at root is a stub; no shared rate-limit store yet (password reset shipped 2026-09-26) |
| Documentation (5) | README (setup, usage, API, troubleshooting), `docs/design.md` (architecture diagrams, sequence, decisions), this tally, demo script, `docs/deployment_aws.md` (real deployment runbook, incl. a "what actually happened" section documenting the real issues hit and fixed), inline module docstrings | 5 | None significant |
| Testing (5) | 1,308 backend tests (99% coverage, 6,035 lines) incl. 29 end-to-end scenarios and edge cases; 46 frontend tests; live benchmark | 5 | Playwright not installed; real-browser check done manually |

## Bonus (up to 10)
Voice interface (browser + Whisper/TTS with visible fallback), advanced portfolio analytics (look-through diversification, mismatch notes, what-if), novel AI techniques: grounding check that blocks ungrounded figures, semantic cache with wrong-hit guards, PII fail-closed pipeline with masked tracing, transparency labels for cached/stale/sample data. Est. **8–10**.

## Weighted score (level-based, the rubric's own method)

The rubric scores each criterion at one of four levels: Excellent 100%, Good 75%, Satisfactory 50%, Needs improvement 25%, then multiplies by the criterion weight. Two readings are shown: **Expected** (what the evidence supports) and **Conservative** (one level lower wherever a real gap could cost a level).

| Category | Criterion | Weight | Expected level | Expected pts | Conservative level | Conservative pts |
|---|---|---:|---|---:|---|---:|
| Technical | Multi-agent architecture | 10 | Excellent | 10.00 | Excellent | 10.00 |
| Technical | LangGraph workflow | 10 | Excellent | 10.00 | Good | 7.50 |
| Technical | RAG implementation | 8 | Excellent | 8.00 | Good | 6.00 |
| Technical | Real-time data integration | 7 | Excellent | 7.00 | Excellent | 7.00 |
| Technical | MCP server | 5 | Excellent | 5.00 | Excellent | 5.00 |
| | **Technical subtotal** | **40** | | **40.00** | | **35.50** |
| UX | Web interface | 10 | Excellent | 10.00 | Good | 7.50 |
| UX | Conversational flow | 8 | Excellent | 8.00 | Good | 6.00 |
| UX | Data visualization | 7 | Excellent | 7.00 | Good | 5.25 |
| | **UX subtotal** | **25** | | **25.00** | | **18.75** |
| Domain | Educational content | 8 | Excellent | 8.00 | Good | 6.00 |
| Domain | Portfolio analysis | 7 | Excellent | 7.00 | Good | 5.25 |
| Domain | Market intelligence | 5 | Excellent | 5.00 | Good | 3.75 |
| | **Domain subtotal** | **20** | | **20.00** | | **15.00** |
| Code | Code organization | 5 | Excellent | 5.00 | Good | 3.75 |
| Code | Documentation | 5 | Excellent | 5.00 | Excellent | 5.00 |
| Code | Testing | 5 | Excellent | 5.00 | Excellent | 5.00 |
| | **Code subtotal** | **15** | | **15.00** | | **13.75** |
| | **CORE TOTAL (of 100)** | **100** | | **100.00** | | **83.00** |
| Bonus | Innovation (up to 10) | 10 | Excellent (9) | 9.00 | Good (7.5) | 7.50 |
| | **TOTAL WITH BONUS** | | | **≈109** | | **≈90.5** |

Why each level is not higher, in the rubric's own terms:
- **Data visualization and market intelligence** were the two criteria at Good; both were improved (interactive charts with tooltips, legends, overlays and a target line; benchmark comparison, segment overview and computed interpretation) and now assess as Excellent in the expected reading. In the conservative reading they stay at Good: the charts are hand-built SVG without zoom/pan, and the interpretation is rule-based and descriptive by design (no news or fundamentals).
- **Conservative one-level drops**: LangGraph ("flawless… advanced state management": own SQLite memory instead of a checkpointer); RAG (was: 90 of 230 articles had placeholder sources, softened by the source-preference ranking — all 230 now carry real sources as of 2026-09-25); Web interface and conversational flow (only lightly checked on small screens; context is the recent turns, not a long-chat summary); Educational content and portfolio analysis (content accuracy is as supplied; no benchmark comparison); Code organization ("production-ready": a legacy client and a stub root `main.py` — though the app is now genuinely deployed and running in production on AWS, which is real evidence for this criterion, not just a claim, and password reset shipped 2026-09-26). MCP is no longer on this list — confirmed working end to end inside Claude Desktop itself as of 2026-09-26.

The placeholder-source gap is fully closed (all 230 KB articles carry real, individually-verified citations); a portfolio-versus-benchmark comparison, login, password reset (email-based, single-use time-limited link, tested locally end-to-end 2026-09-26), a real production deployment (AWS EC2 + Docker + Caddy, HTTPS, survives reboot), and MCP-in-Claude-Desktop are all done. The "shared rate-limit store" gap is also less relevant now than when Vercel (multi-instance/stateless) was the deployment target — a single EC2 process's in-memory limiter is a reasonable fit for the actual, current deployment topology. The remaining conservative-reading gaps (LangGraph checkpointer, a legacy embeddings client, a stub root `main.py`) are minor and mostly a matter of taste rather than missing functionality.

## Requirement checklist (problem statement §3)

| # | Requirement | Status |
|---|---|---|
| 1 | All six agents, 80%+ coverage, fallbacks | Done (99%) |
| 2 | Chat + portfolio dashboard + market overview | Done |
| 3 | 50–100 articles, category filter, attribution | Done (230) |
| 4 | Live quotes, caching, rate limits, trends | Done |
| 5 | MCP (optional) | Done (`docs/mcp.md`) |
| 6 | Exact + semantic cache for Q&A | Done |
| 7 | Keyword + vector search via Qdrant | Done |
| 8 | Audio question and answer | Done |
| 9 | Python logger | Done (JSON) |
| 10 | Langfuse | Done (+ LangSmith) |
| 11 | MacBook + Vercel | MacBook done; **deployed and live** at `https://clarityfi.rajur-dev.com` (2026-09-26) — AWS EC2 + Docker + Caddy was used instead of Vercel, a deliberate choice since the backend's embedded Qdrant index and local SQLite need a persistent disk, which doesn't fit Vercel's serverless model without re-architecting to hosted Qdrant/Postgres (see `docs/deployment_aws.md`) |
| 12 | Docker image | **Done** — `Dockerfile` + `docker-compose.yml` built, running in production on EC2, `restart: unless-stopped`, survives reboot |
| 13 | Synthetic portfolios, 10 users | Done |
| 14 | End-to-end tests | Done |
| 15 | Session storage (SQLite) | Done |
| 16 | QA data provided | Done (ingested) |
| 17 | UI reference | Done (provided React code kept and wired) |
| 18 | Presidio on LLM input/output | Done |
| 19 | Existing code reused | Done |
| 20 | Reconciled against rubric | This document |
