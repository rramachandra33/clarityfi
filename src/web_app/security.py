"""Request security shared by every route: login required, ownership of `/{user_id}` paths, CSRF header.

`require_login` is installed as an application-wide dependency, so a new route is protected by default; only the
paths in PUBLIC_PATHS are open. Handlers read the caller with `principal_of(request)`.
"""
from __future__ import annotations

from fastapi import Request, Response

from src.core.config import get_config
from src.web_app.auth_service import AuthService, Principal
from src.web_app.errors import ApiError

CSRF_HEADER = "x-clarity-client"
CSRF_VALUE = "web"
PUBLIC_PATHS = frozenset({"/api/health", "/api/auth/login", "/api/auth/register", "/api/auth/logout", "/api/auth/me", "/api/auth/config",
                          "/api/auth/forgot-password", "/api/auth/reset-password",
                          "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def session_token(request: Request) -> str | None:
    return request.cookies.get(get_config().auth.cookie_name)


def set_session_cookie(request: Request, response: Response, token: str) -> None:
    cfg = get_config().auth
    response.set_cookie(cfg.cookie_name, token, max_age=cfg.session_days * 86400, httponly=True, samesite="lax",
                        secure=request.url.scheme == "https", path="/")


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(get_config().auth.cookie_name, path="/")


def principal_of(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if principal is None:  # only reachable if a route were exempted from require_login by mistake: fail closed
        raise ApiError(401, "Please sign in.", "not_authenticated")
    return principal


def require_login(request: Request) -> None:
    """Application-wide dependency: CSRF check, sign-in check, and ownership of any `{user_id}` in the path."""
    if request.method not in SAFE_METHODS and request.headers.get(CSRF_HEADER) != CSRF_VALUE:
        raise ApiError(403, "This request was blocked because it did not come from the ClarityFI app.", "csrf_blocked")
    path = request.url.path
    if path in PUBLIC_PATHS or (path == "/api/glossary" and request.method == "GET"):
        return
    service: AuthService = request.app.state.auth
    principal = service.principal_for(session_token(request))
    if principal is None:
        raise ApiError(401, "Please sign in.", "not_authenticated")
    request.state.principal = principal
    target = request.path_params.get("user_id")
    if target is not None and not principal.owns(target):
        raise ApiError(403, "You can only access your own data.", "forbidden")


def require_admin(request: Request) -> Principal:
    principal = principal_of(request)
    if not (principal.is_admin or principal.any_user):
        raise ApiError(403, "Only an administrator can do this.", "admin_only")
    return principal
