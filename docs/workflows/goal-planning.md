# Goal Planning — UI → backend workflow

## Example used

Goal: **"House"**, target **$50,000**, current savings **$5,000**, monthly contribution **$300**, **8 years**.

Source test: `tests/unit/test_api_routes.py::test_goal_projection_variants`

```python
body = {"name": "House", "target_amount": 50000, "current_savings": 5000, "monthly_contribution": 300, "years": 8}
r = env.post("/api/goals/project", json=body).json()
# 3 scenarios centered on assumed_return_pct (0 by default)
assert len(r["projection"]["scenarios"]) == 3 and r["explanation"] is None and r["assumptions"]
assert r["projection"]["scenarios"][1]["label"] == "Your assumption" and r["projection"]["scenarios"][1]["annual_return"] == 0.0
assert env.post("/api/goals/project", json={**body, "explain": True}).json()["explanation"]
```

**Important architectural note:** like Portfolio Analysis, this tab does **not** go through
`/api/chat`/the router. Clicking "Project my goal" is fully deterministic (no LLM). Only "Project and
explain" (`explain=true`) adds one LLM call, and it calls `GoalPlanningAgent.arun()` directly with the
goal already attached to the request — the LLM-based *extraction* code path in `GoalPlanningAgent`
(parsing figures out of free text) is **not** used here at all, because the UI form already gives the
agent a structured goal. That extraction path only runs when a user types something like "I want $50,000
in 8 years" into the Finance Q&A chat box.

---

## 1. User action in the browser

The user fills in the goal-builder form (name, target amount, saved so far, monthly contribution, years, and an
"Assumed return (%)" field defaulting to 0) and clicks **"Project and explain"**.

| Step | Function | File |
|---|---|---|
| Tab mounted | `GoalPlanning({ userId })` | `frontend/src/tabs/Goals.jsx` |
| Form state | `set(key)` | `frontend/src/tabs/Goals.jsx` |
| Button click | `project(explain=true)` | `frontend/src/tabs/Goals.jsx` |

## 2. Frontend request

```js
api('/api/goals/project', {
  method: 'POST',
  body: {
    name: 'House', target_amount: 50000, current_savings: 5000,
    monthly_contribution: 300, years: 8, assumed_return_pct: 0, user_id: userId, explain: true,
  },
})
```

`api()` (`frontend/src/api.js`) wraps `apiFetch()` (`frontend/src/http.js`), which adds the
`X-Clarity-Client: web` header (required for this POST) and the session cookie.

## 3. FastAPI request pipeline

`request_context` middleware → `CORSMiddleware` (**[EXTERNAL]** Starlette) → `require_login()`
(`src/web_app/security.py`) — `/api/goals/project` has no `{user_id}` path segment, so ownership of a
specific portfolio isn't checked here; instead the **route itself** ignores whatever `user_id` was sent
in the body and substitutes the caller's own id (see step 4.2) so a user cannot get a projection narrated
"as" someone else. All in `src/web_app/api/main.py` / `security.py`.

## 4. Route handler

`project(body, request, svc)` in `src/web_app/api/routes_goals.py`:

1. `enforce(svc.limiters["expensive" if body.explain else "read"], client_key(request))` (`src/web_app/ratelimit.py`).
2. `principal_of(request)` (`src/web_app/security.py`) then `owner = ... else caller.user_id` — the signed-in user's own id always wins over anything in the request body.
3. `UserRepository.get(owner)` (`src/data/repositories.py`) supplies a default risk tolerance if the request didn't state one.
4. `_goal_from(body, risk)` (same file) builds a `Goal` model (`src/core/models.py`) from the form fields — `target_date` is computed from `years` via `today + timedelta(days=round(years * 365.25))` if a date wasn't given directly.
5. `project_goal(goal, today=date.today())` — the deterministic projection, always run (see step 5).
6. `GoalPlanningAgent._facts(goal, projection, [])["assumptions"]` (`src/agents/goal_planning.py`) — reused as a static method purely to build the human-readable assumption strings, no LLM involved.
7. Only if `explain=true`: `svc.sanitize(goal.name, max_chars=80)` (`src/web_app/api/services.py`) PII-redacts the free-text goal name before it can reach a model, then `svc.goal_agent.arun(AgentRequest(question="Explain my goal projection in plain language.", user_id=owner, context={"goal": {...sanitized name...}}))`.
8. Returns a `ProjectionOut` (`src/web_app/api/models.py`).

## 5. `project_goal()` — the deterministic engine

`project_goal(goal, today)` in `src/core/projections.py` (a **pure function**, no I/O):

- Computes `years = (target_date - today).days / 365.25`.
- For each of three scenarios — below / your assumption / above, centered on the user's own `goal.assumed_return_pct` (0% by default) and spread by +/- `SPREAD_PCT` = 2 percentage points — it compounds `current_savings` plus `monthly_contribution` at that rate, checks `on_track` against `target_amount`, computes the `shortfall`, the `required_monthly` amount needed to reach the target from here, and `years_to_goal` at the current pace. `goal.risk_tolerance` is not consulted here; it still matters elsewhere (e.g. the Portfolio agent's stated-risk comparison).
- `yearly_path()` (same file) produces a year-by-year value series for the chart, and each scenario also gets a `real_projected_value` discounted by `DEFAULT_INFLATION` (2.5%/year) — "today's dollars".

Because the House example asks for exactly one of `target_date`/`years`, `GoalIn`'s validator
(`src/web_app/api/models.py`) would reject a body that gave both or neither — the second half of the same
test checks exactly that (`422` for `target_date` and `years` together, and for neither).

## 6. Only when `explain=true`: the narration

`GoalPlanningAgent.arun()` (`BaseAgent`, `src/agents/base.py`) logs `Entering goal_planning agent`, then
`GoalPlanningAgent._execute()` (`src/agents/goal_planning.py`):

1. Since `request.context["goal"]` is already set (the route provided it), the agent takes the **"provided" branch**: `goal = Goal.model_validate(given)`, `source="provided"`. It does **not** call `self._extract()` — the LLM-based figure-extraction code path (`with_structured_output(GoalParams)`, same file) is skipped entirely for this tab.
2. `project_goal(goal, today=self._today())` — recomputed here (the agent doesn't reuse the route's projection object, since it's called as a standalone unit).
3. `self._facts(goal, projection, [])` builds the same `facts` dict shape used everywhere: goal name/amount/date, per-scenario numbers, assumptions.
4. `narrate(request.question, facts, fallback_template, agent_name="goal_planning")` in `src/agents/narration.py`:
   - `ainvoke_chat(..., purpose="goal_planning narration")` (`src/core/llm.py`) → **[EXTERNAL]** OpenAI Chat Completions API via `langchain_openai.ChatOpenAI`.
   - `ungrounded_numbers(text, facts)` (`src/agents/narration.py`) rejects the narration (falling back to `GoalPlanningAgent._template(facts)`, a deterministic string) if the model states a dollar/percent figure not present in `facts`.
5. Logs `Leaving goal_planning agent after N ms (...)`.

## 7. Response back to the browser

`ProjectionOut` (`src/web_app/api/models.py`): `goal`, `projection` (three `scenarios` + `yearly_path`),
`assumptions`, `on_track_expected`, and `explanation` (only when `explain=true`; `explanation_error` if the
narration step failed but the deterministic numbers are still returned).

## 8. Rendering

`GoalPlanning` (`frontend/src/tabs/Goals.jsx`) renders the scenario table, the yearly-path chart with a
dashed target line (`LineChart`, `frontend/src/common.jsx`), the assumption list, and — if present — the
"In plain language" explanation box.

---

## Call chain at a glance

```
GoalPlanning.project(explain=true)                  frontend/src/tabs/Goals.jsx   — sends the goal form to the backend and stores the projection result
  → api('/api/goals/project')                       frontend/src/api.js   — makes the request and turns a non-2xx response into a thrown error
    → apiFetch()                                    frontend/src/http.js   — adds the CSRF header and session cookie to the request
      → fetch()                                     [EXTERNAL] browser   — the browser's network call
        → require_login()                           src/web_app/security.py   — rejects the request unless a valid session cookie is present
        → project()                                 src/web_app/api/routes_goals.py   — the /api/goals/project route handler
          → enforce()                                src/web_app/ratelimit.py   — enforces the rate limit for this route
          → principal_of()                           src/web_app/security.py   — gets the signed-in caller, so the goal is always projected as them
          → _goal_from()                             src/web_app/api/routes_goals.py   — builds a Goal from the submitted form fields
          → project_goal()                           src/core/projections.py   (pure function)   — computes three scenarios (below/your assumption/above) centered on the user's own assumed return
          → svc.sanitize()                           src/web_app/api/services.py   — validates and redacts the free-text goal name
            → redact_text()                           src/core/pii.py          [EXTERNAL: presidio, spaCy]   — strips personal data from the goal name
          → GoalPlanningAgent.arun() → _execute()    src/agents/goal_planning.py   — narrates the already-computed projection
            → project_goal()                          src/core/projections.py   (pure function, again)   — recomputes the same scenarios inside the agent
            → narrate()                               src/agents/narration.py   — asks the model to explain the numbers, then checks its reply for invented figures
              → ainvoke_chat()                         src/core/llm.py   — calls the chat model and returns its plain-text reply
                → ChatOpenAI.ainvoke()                   [EXTERNAL: OpenAI API]   — the actual chat-completion request to OpenAI
              → ungrounded_numbers()                   src/agents/narration.py  (grounding check)   — rejects the narration if it states a figure not in the computed facts
          → ProjectionOut(...)                        src/web_app/api/models.py   — shapes the final JSON sent back to the browser
      ← JSON response
  ← scenario table + chart + explanation box           frontend/src/tabs/Goals.jsx   — renders the result
```
