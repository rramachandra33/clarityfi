"""User profile endpoints. Accounts are created through /api/auth/register; every route here needs a signed-in user."""
from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from src.core.models import UserProfile
from src.data.db import Database
from src.data.repositories import UserRepository
from src.web_app.api.models import EmailUpdate, ProfileUpdate
from src.web_app.errors import ApiError


def build_users_router(get_db: Callable[[], Database]) -> APIRouter:
    router = APIRouter(prefix="/api/users", tags=["users"])

    @router.get("", response_model=list[UserProfile])
    def list_demo_users(db: Database = Depends(get_db)) -> list[UserProfile]:
        """The 10 synthetic demo users (registered real ids are not listed)."""
        return UserRepository(db).list(synthetic_only=True)

    @router.put("/{user_id}/profile", response_model=UserProfile, summary="Update a user's risk tolerance, experience level or horizon")
    def update_profile(user_id: str, body: ProfileUpdate, db: Database = Depends(get_db)) -> UserProfile:
        profile = UserRepository(db).update_profile(user_id, risk_tolerance=body.risk_tolerance, experience_level=body.experience_level,
                                                    horizon_years=body.horizon_years)
        if profile is None:
            raise HTTPException(status_code=404, detail="Unknown user id")
        return profile

    @router.get("/{user_id}", response_model=UserProfile)
    def get_user(user_id: str, db: Database = Depends(get_db)) -> UserProfile:
        profile = UserRepository(db).get(user_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Unknown user id")
        return profile

    @router.put("/{user_id}/email", response_model=UserProfile, summary="Set or remove the email used for password reset",
                responses={409: {"description": "That email is already used by another account"}})
    def update_email(user_id: str, body: EmailUpdate, db: Database = Depends(get_db)) -> UserProfile:
        clean_email = None
        if body.email and body.email.strip():
            try:
                clean_email = UserProfile(user_id=user_id, email=body.email).email
            except ValidationError as exc:
                raise ApiError(422, "Enter a valid email address.", "invalid_email") from exc
        if not UserRepository(db).set_email(user_id, clean_email):
            raise ApiError(409, "That email is already used by another account.", "email_taken")
        return UserRepository(db).get(user_id)

    return router
