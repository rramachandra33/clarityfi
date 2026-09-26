"""Goal endpoints: save/list/delete a user's goals, and project any goal (deterministic; explanation optional)."""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Callable

from fastapi import APIRouter, Depends, Path, Request, Response

from src.agents.base import AgentRequest
from src.agents.goal_planning import GoalPlanningAgent
from src.core.models import Goal, RiskTolerance
from src.core.projections import project_goal
from src.data.repositories import GoalRepository, UserRepository
from src.web_app.api.models import NOT_FOUND, TOO_MANY, GoalIn, GoalOut, ProjectionIn, ProjectionOut, UserIdPath
from src.web_app.api.services import ApiServices
from src.web_app.errors import ApiError
from src.web_app.security import principal_of
from src.web_app.ratelimit import client_key, enforce

MAX_SAVED_GOALS = 20


def _goal_from(body: GoalIn, default_risk: RiskTolerance, today: date | None = None) -> Goal:
    today = today or date.today()
    target = body.target_date or today + timedelta(days=round((body.years or 0) * 365.25))
    return Goal(name=body.name, target_amount=body.target_amount, target_date=target, current_savings=body.current_savings,
                monthly_contribution=body.monthly_contribution, risk_tolerance=body.risk_tolerance or default_risk,
                assumed_return_pct=body.assumed_return_pct)


def build_goals_router(get_services: Callable[[], ApiServices]) -> APIRouter:
    router = APIRouter(prefix="/api/goals", tags=["goals"])

    # NOTE: /project is declared BEFORE the /{user_id} routes so it is never captured as a user id.
    @router.post("/project", response_model=ProjectionOut, responses=TOO_MANY, summary="Project a savings goal under three scenarios",
                 description="Deterministic compound-growth projection: a below/expected/above trio centered on your own `assumed_return_pct` "
                             "(0% by default; illustrations, not forecasts). Give exactly one of `target_date` or `years`. `explain=true` adds a narration.")
    async def project(body: ProjectionIn, request: Request, svc: ApiServices = Depends(get_services)) -> ProjectionOut:
        enforce(svc.limiters["expensive" if body.explain else "read"], client_key(request))
        risk = body.risk_tolerance
        caller = principal_of(request)
        owner = body.user_id if caller.any_user else caller.user_id  # a signed-in user is always the owner; the body id is not trusted
        if risk is None and owner:
            profile = UserRepository(svc.db).get(owner)
            risk = profile.risk_tolerance if profile else None
        goal = _goal_from(body, risk or RiskTolerance.moderate)
        projection = project_goal(goal, today=date.today())
        assumptions = GoalPlanningAgent._facts(goal, projection, [])["assumptions"]
        explanation = error = None
        if body.explain:
            safe_name = await svc.sanitize(goal.name, max_chars=80)  # the goal name is free text that would reach the LLM
            result = await svc.goal_agent.arun(AgentRequest(
                question="Explain my goal projection in plain language.", user_id=owner or "anonymous",
                context={"goal": goal.model_copy(update={"name": safe_name}).model_dump(mode="json")}))
            explanation, error = (result.content, None) if result.ok else (None, result.error)
        return ProjectionOut(goal=goal, projection=projection, assumptions=assumptions,
                             on_track_expected=projection.scenarios[1].on_track, explanation=explanation, explanation_error=error)

    @router.get("/{user_id}", response_model=list[GoalOut], responses=NOT_FOUND, summary="A user's saved goals")
    def list_goals(user_id: UserIdPath, request: Request, svc: ApiServices = Depends(get_services)) -> list[GoalOut]:
        enforce(svc.limiters["read"], client_key(request))
        profile = svc.require_user(user_id)
        return [GoalOut(id=i, goal=g) for i, g in GoalRepository(svc.db).list_rows(profile.user_id)]

    @router.post("/{user_id}", response_model=GoalOut, status_code=201, responses={**NOT_FOUND, 409: {"description": "Too many saved goals"}},
                 summary="Save a goal for a user",
                 description="Creates a new saved goal, unless this user already has one with the same name (case/whitespace-insensitive) "
                             "— in that case, the existing goal is updated in place (same id, response status 200) instead of adding a duplicate row.")
    def save_goal(user_id: UserIdPath, body: GoalIn, request: Request, response: Response, svc: ApiServices = Depends(get_services)) -> GoalOut:
        enforce(svc.limiters["read"], client_key(request))
        profile = svc.require_user(user_id)
        repo = GoalRepository(svc.db)
        goal = _goal_from(body, profile.risk_tolerance)
        existing_id = repo.find_id_by_name(profile.user_id, goal.name)
        if existing_id is not None:
            repo.update(profile.user_id, existing_id, goal)
            response.status_code = 200
            return GoalOut(id=existing_id, goal=goal)
        if repo.count(profile.user_id) >= MAX_SAVED_GOALS:
            raise ApiError(409, f"You can save up to {MAX_SAVED_GOALS} goals", "too_many_goals")
        return GoalOut(id=repo.add(profile.user_id, goal), goal=goal)

    @router.delete("/{user_id}/{goal_id}", status_code=204, responses=NOT_FOUND, summary="Delete a saved goal")
    def delete_goal(user_id: UserIdPath, goal_id: int = Path(ge=1), request: Request = None, svc: ApiServices = Depends(get_services)) -> None:
        enforce(svc.limiters["read"], client_key(request))
        profile = svc.require_user(user_id)
        if not GoalRepository(svc.db).delete(profile.user_id, goal_id):
            raise ApiError(404, "No such goal for this user", "goal_not_found")

    return router
