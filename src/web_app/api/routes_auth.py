"""Sign in / sign up / sign out. The session lives in an HttpOnly cookie; the server keeps only a hash of the token."""
from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from src.core.config import get_config, settings
from src.core.models import UserProfile
from src.web_app.auth_service import AuthService
from src.web_app.errors import ApiError
from src.web_app.security import clear_session_cookie, client_ip, session_token, set_session_cookie

DEMO_USERS = [f"learner-{n}" for n in range(101, 111)]


class Credentials(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=256)


class RegisterIn(Credentials):
    email: str | None = Field(default=None, max_length=254, description="Optional; enables self-service password reset")


class ForgotPasswordIn(BaseModel):
    email: str = Field(min_length=1, max_length=254)


class ResetPasswordIn(BaseModel):
    token: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class Me(BaseModel):
    user_id: str
    synthetic: bool
    is_admin: bool
    profile: UserProfile


class AuthConfigOut(BaseModel):
    min_password_length: int
    demo_login: bool
    demo_users: list[str] = Field(default_factory=list)
    demo_password: str | None = Field(default=None, description="Shown only when demo login is on: it opens synthetic demo users only")


def build_auth_router() -> APIRouter:
    router = APIRouter(prefix="/api/auth", tags=["auth"])

    def service(request: Request) -> AuthService:
        return request.app.state.auth

    def me(svc: AuthService, uid: str, is_admin: bool) -> Me:
        profile = svc.users.get(uid)
        return Me(user_id=profile.user_id, synthetic=svc.users.is_synthetic(uid), is_admin=is_admin, profile=profile)

    @router.get("/config", response_model=AuthConfigOut, summary="What the sign-in screen needs to know")
    def config() -> AuthConfigOut:
        demo = settings.demo_login_enabled
        return AuthConfigOut(min_password_length=get_config().auth.min_password_length, demo_login=demo,
                             demo_users=DEMO_USERS if demo else [], demo_password=settings.demo_password if demo else None)

    @router.post("/register", response_model=Me, status_code=201, summary="Create an account and sign in",
                 description="A new user id is saved with an empty portfolio. Ids that already have a password (and the demo ids) are taken. "
                             "An optional email enables self-service password reset later.")
    def register(body: RegisterIn, request: Request, response: Response) -> Me:
        svc = service(request)
        profile, token = svc.register(body.user_id, body.password, body.email)
        set_session_cookie(request, response, token)
        return me(svc, profile.user_id, is_admin=svc.principal_for(token).is_admin)

    @router.post("/login", response_model=Me, summary="Sign in", responses={401: {"description": "Incorrect user id or password"},
                 429: {"description": "Too many failed attempts (see Retry-After)"}})
    def login(body: Credentials, request: Request, response: Response) -> Me:
        svc = service(request)
        profile, token = svc.login(body.user_id, body.password, client_ip(request))
        set_session_cookie(request, response, token)
        return me(svc, profile.user_id, is_admin=svc.principal_for(token).is_admin)

    @router.post("/logout", status_code=204, summary="Sign out (ends this session)")
    def logout(request: Request, response: Response) -> Response:
        service(request).logout(session_token(request))
        clear_session_cookie(response)
        response.status_code = 204
        return response

    @router.get("/me", response_model=Me, summary="Who am I? (401 when not signed in)")
    def whoami(request: Request) -> Me:
        svc = service(request)
        principal = svc.principal_for(session_token(request))
        if principal is None:
            raise ApiError(401, "Please sign in.", "not_authenticated")
        return me(svc, principal.user_id, principal.is_admin)

    @router.post("/forgot-password", status_code=204, summary="Request a password-reset email",
                 description="Always answers the same way, whether or not the email is on an account, so this cannot be used to "
                             "discover which addresses have accounts. Only works for accounts that have an email on file.",
                 responses={429: {"description": "Too many requests (see Retry-After)"}})
    def forgot_password(body: ForgotPasswordIn, request: Request) -> None:
        service(request).request_password_reset(body.email, client_ip(request))

    @router.post("/reset-password", status_code=204, summary="Set a new password from a reset-email token",
                 responses={400: {"description": "The token is unknown, expired, or already used"}})
    def reset_password(body: ResetPasswordIn, request: Request) -> None:
        if not service(request).reset_password(body.token, body.new_password):
            raise ApiError(400, "This reset link is invalid or has expired. Request a new one.", "invalid_reset_token")

    return router
