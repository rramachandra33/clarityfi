"""Sign-in: hashing, register/login/logout, sessions, throttling, CSRF, route protection and per-user ownership."""
import pytest
from fastapi.testclient import TestClient

from src.agents.goal_planning import GoalPlanningAgent
from src.agents.portfolio import PortfolioAgent
from src.core import auth
from src.core.config import settings
from src.data.db import Database
from src.data.repositories import AuthRepository, UserRepository
from src.data.seed import seed
from src.web_app import auth_service as auth_mod
from src.web_app.api import main
from src.web_app.api.services import ApiServices, default_limiters
from src.web_app.auth_service import AuthService, Principal
from src.web_app.security import PUBLIC_PATHS
from tests.unit._agent_fakes import FakeLLM, FakeMarket, quote

pytestmark = pytest.mark.real_auth
HEADERS = {"X-Clarity-Client": "web"}
GOOD = "a-decent-password"


class FakeEmailSender:
    """Captures messages instead of sending them, so tests can extract the reset token from the body."""

    def __init__(self, error=None):
        self.sent, self.error = [], error

    def send(self, message):
        if self.error is not None:
            raise self.error
        self.sent.append(message)


def token_from(message) -> str:
    return message.text_body.split("token=", 1)[1].split()[0]


# ------------------------------------------------------------------------------ hashing and tokens
def test_password_hash_is_salted_verifiable_and_rejects_wrong_or_malformed():
    h1, h2 = auth.hash_password(GOOD), auth.hash_password(GOOD)
    assert h1 != h2 and h1.startswith("scrypt$") and GOOD not in h1
    assert auth.verify_password(GOOD, h1) and not auth.verify_password("wrong-password", h1)
    for bad in (None, "", "garbage", "scrypt$x$y$z$a$b", "bcrypt$1$2$3$YQ==$YQ=="):
        assert auth.verify_password(GOOD, bad) is False


def test_hashes_written_with_other_parameters_still_verify(monkeypatch):
    monkeypatch.setattr(auth, "SCRYPT_N", 2 ** 11)
    h = auth.hash_password(GOOD)
    monkeypatch.setattr(auth, "SCRYPT_N", 2 ** 10)
    assert "$2048$" in h and auth.verify_password(GOOD, h)


@pytest.mark.parametrize("pw,fragment", [("short", "at least 8"), ("x" * 129, "at most 128"), (" leading-space1", "space"), ("tab\tinside-pw", "control"), (123, "Enter a password")])
def test_password_policy(pw, fragment):
    from src.core.errors import GuardrailError
    with pytest.raises(GuardrailError, match=fragment):
        auth.validate_password(pw)
    assert auth.validate_password("correct horse battery staple") == "correct horse battery staple"


def test_tokens_are_random_and_only_hashes_are_comparable():
    a, b = auth.new_token(), auth.new_token()
    assert a != b and len(a) >= 40 and auth.token_hash(a) != a and auth.token_hash(a) == auth.token_hash(a)


# --------------------------------------------------------------------------------------------- app
@pytest.fixture
def env(tmp_path, monkeypatch):
    db = Database(tmp_path / "auth.db")
    db.init_schema()
    seed(db)
    email = FakeEmailSender()
    service = AuthService(db, email_sender=email)
    service.ensure_demo_passwords()
    market = FakeMarket({"VTI": quote("VTI", 300.0), "VXUS": quote("VXUS", 60.0), "BND": quote("BND", 75.0)})
    llm = FakeLLM("ok")
    svc = ApiServices(db=db, market=market, news=None, portfolio_agent=PortfolioAgent(db, market, llm_factory=lambda: llm, callbacks_factory=lambda: []),
                      goal_agent=GoalPlanningAgent(db, llm_factory=lambda: llm, callbacks_factory=lambda: []), news_agent=None, memory=None,
                      limiters=default_limiters())
    monkeypatch.setattr(main.app.state, "auth", service)
    monkeypatch.setattr(main, "api_services", svc)
    main.app.dependency_overrides[main.get_services] = lambda: svc
    main.app.dependency_overrides[main.get_db] = lambda: db
    with TestClient(main.app, headers=HEADERS) as client:
        client.db, client.service, client.email = db, service, email
        yield client
    main.app.dependency_overrides.clear()


def register(client, uid="new-learner", pw=GOOD, email=None):
    body = {"user_id": uid, "password": pw}
    if email is not None:
        body["email"] = email
    return client.post("/api/auth/register", json=body)


def login(client, uid, pw=GOOD):
    return client.post("/api/auth/login", json={"user_id": uid, "password": pw})


def forgot(client, email):
    return client.post("/api/auth/forgot-password", json={"email": email})


def reset(client, token, new_pw=GOOD):
    return client.post("/api/auth/reset-password", json={"token": token, "new_password": new_pw})


def test_register_creates_user_signs_in_and_sets_a_locked_down_cookie(env):
    r = register(env)
    assert r.status_code == 201 and r.json()["user_id"] == "new-learner" and r.json()["synthetic"] is False and r.json()["is_admin"] is False
    cookie = r.headers["set-cookie"].lower()
    assert "clarity_session=" in cookie and "httponly" in cookie and "samesite=lax" in cookie and "path=/" in cookie and "secure" not in cookie
    assert env.get("/api/auth/me").json()["user_id"] == "new-learner"
    assert env.get("/api/portfolio/new-learner/analysis").json()["empty"] is True


def test_cookie_is_secure_over_https(env):
    r = TestClient(main.app, base_url="https://testserver", headers=HEADERS).post("/api/auth/register", json={"user_id": "https-user", "password": GOOD})
    assert "secure" in r.headers["set-cookie"].lower()


@pytest.mark.parametrize("uid,pw,code,status", [("ab", GOOD, "invalid_user_id", 422), ("bad;id", GOOD, "invalid_user_id", 422),
                                                 ("valid-id", "short", "weak_password", 422)])
def test_register_validation(env, uid, pw, code, status):
    r = register(env, uid, pw)
    assert r.status_code == status and r.json()["code"] == code and "set-cookie" not in r.headers


def test_taken_ids_demo_ids_and_case_variants_cannot_be_registered(env):
    register(env, "taken-id")
    env.cookies.clear()
    assert register(env, "TAKEN-ID").json()["code"] == "user_id_taken"
    assert register(env, "learner-105").status_code == 409  # demo ids sign in with the demo password


def test_an_id_from_before_sign_in_existed_can_be_claimed_once(env):
    UserRepository(env.db).ensure("legacy-user")
    assert register(env, "legacy-user").status_code == 201
    env.cookies.clear()
    assert register(env, "legacy-user", "another-password").status_code == 409


def test_login_success_wrong_password_and_unknown_user_look_identical(env):
    register(env)
    env.cookies.clear()
    bad_pw, no_user = login(env, "new-learner", "wrong-password"), login(env, "ghost-user")
    assert bad_pw.status_code == no_user.status_code == 401 and bad_pw.json() == no_user.json() and bad_pw.json()["code"] == "invalid_credentials"
    assert "set-cookie" not in bad_pw.headers
    ok = login(env, "NEW-LEARNER")
    assert ok.status_code == 200 and ok.json()["user_id"] == "new-learner"  # id is case-insensitive, canonical spelling returned


def test_demo_users_sign_in_with_the_demo_password(env):
    r = login(env, "learner-105", settings.demo_password)
    assert r.status_code == 200 and r.json()["synthetic"] is True
    assert login(env, "learner-105", "not-the-demo-password").status_code == 401


def test_repeated_failures_are_throttled_with_retry_after_and_do_not_lock_out_others(env):
    register(env)
    env.cookies.clear()
    for _ in range(5):
        assert login(env, "new-learner", "wrong-password").status_code == 401
    blocked = login(env, "new-learner", GOOD)  # even the right password waits out the window
    assert blocked.status_code == 429 and blocked.json()["code"] == "too_many_attempts" and int(blocked.headers["Retry-After"]) >= 1
    assert login(env, "learner-105", settings.demo_password).status_code == 200  # another id from the same client is fine


def test_per_client_limit_stops_guessing_across_many_ids(env, monkeypatch):
    monkeypatch.setattr(env.service.by_client, "max_calls", 3)
    for i in range(3):
        assert login(env, f"guess-{i}-user", "whatever-pass").status_code == 401
    assert login(env, "learner-105", settings.demo_password).status_code == 429


def test_logout_ends_the_session_server_side(env):
    register(env)
    token = env.cookies.get("clarity_session")
    assert env.post("/api/auth/logout").status_code == 204
    assert env.get("/api/auth/me").status_code == 401
    env.cookies.set("clarity_session", token)  # replaying the old cookie no longer works
    assert env.get("/api/auth/me").status_code == 401


def test_only_a_hash_of_the_token_is_stored_and_each_login_gets_a_new_token(env):
    register(env)
    first = env.cookies.get("clarity_session")
    env.cookies.clear()
    login(env, "new-learner")
    second = env.cookies.get("clarity_session")
    assert first != second
    with env.db.connect() as conn:
        stored = [r[0] for r in conn.execute("SELECT token_hash FROM auth_sessions")]
        pw = conn.execute("SELECT password_hash FROM users WHERE user_id='new-learner'").fetchone()[0]
    assert first not in stored and second not in stored and auth.token_hash(second) in stored and GOOD not in pw


def test_expired_sessions_are_rejected_and_removed(env):
    register(env)
    with env.db.connect() as conn:
        conn.execute("UPDATE auth_sessions SET expires_at = '2000-01-01T00:00:00+00:00'")
    assert env.get("/api/auth/me").status_code == 401
    with env.db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0


def test_a_session_for_a_deleted_user_is_rejected(env):
    register(env)
    repo = AuthRepository(env.db)
    with env.db.connect() as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM users WHERE user_id = 'new-learner'")
    assert env.get("/api/auth/me").status_code == 401 and repo.delete_user_sessions("new-learner") == 1


# --------------------------------------------------------------------------------------- password reset
def test_register_with_email_then_full_reset_flow_works_end_to_end(env):
    register(env, "reset-user", email="reset-user@example.com")
    env.cookies.clear()
    assert forgot(env, "reset-user@example.com").status_code == 204
    assert len(env.email.sent) == 1
    message = env.email.sent[0]
    assert message.to == "reset-user@example.com" and "reset your" in message.subject.lower()
    token = token_from(message)
    assert reset(env, token, "a-new-decent-password").status_code == 204
    assert login(env, "reset-user", "a-new-decent-password").status_code == 200
    assert login(env, "reset-user", GOOD).status_code == 401  # the old password no longer works


def test_reset_invalidates_every_existing_session(env):
    register(env, "session-user", email="session-user@example.com")
    old_cookie = env.cookies.get("clarity_session")
    forgot(env, "session-user@example.com")
    reset(env, token_from(env.email.sent[0]), "a-new-decent-password")
    with env.db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0
    replay = TestClient(main.app, headers=HEADERS, cookies={"clarity_session": old_cookie})
    assert replay.get("/api/auth/me").status_code == 401


def test_forgot_password_answers_identically_whether_or_not_the_email_exists(env):
    register(env, "has-email", email="real@example.com")
    known, unknown = forgot(env, "real@example.com"), forgot(env, "nobody@example.com")
    assert known.status_code == unknown.status_code == 204
    assert len(env.email.sent) == 1  # only the real address actually got a message


def test_forgot_password_is_a_no_op_for_an_account_with_no_email_on_file(env):
    register(env, "no-email-user")  # no email supplied at registration
    assert forgot(env, "no-email-user@example.com").status_code == 204
    assert env.email.sent == []


def test_reset_token_is_single_use_and_rejects_an_unknown_token(env):
    register(env, "single-use", email="single-use@example.com")
    forgot(env, "single-use@example.com")
    token = token_from(env.email.sent[0])
    assert reset(env, token, "a-new-decent-password").status_code == 204
    assert reset(env, token, "another-decent-password").status_code == 400  # already used
    assert reset(env, "not-a-real-token", GOOD).status_code == 400


def test_an_expired_reset_token_is_rejected(env):
    register(env, "expired-case", email="expired-case@example.com")
    forgot(env, "expired-case@example.com")
    token = token_from(env.email.sent[0])
    with env.db.connect() as conn:
        conn.execute("UPDATE password_resets SET expires_at = '2000-01-01T00:00:00+00:00' WHERE user_id = 'expired-case'")
    assert reset(env, token, GOOD).status_code == 400


def test_a_weak_new_password_does_not_burn_the_token(env):
    register(env, "typo-user", email="typo-user@example.com")
    forgot(env, "typo-user@example.com")
    token = token_from(env.email.sent[0])
    weak = reset(env, token, "short")
    assert weak.status_code == 422 and weak.json()["code"] == "weak_password"
    assert reset(env, token, "a-fine-password-now").status_code == 204  # the same token still works


def test_requesting_a_second_reset_invalidates_the_first_token(env):
    register(env, "two-links", email="two-links@example.com")
    forgot(env, "two-links@example.com")
    first_token = token_from(env.email.sent[0])
    forgot(env, "two-links@example.com")
    second_token = token_from(env.email.sent[1])
    assert reset(env, first_token, GOOD).status_code == 400
    assert reset(env, second_token, GOOD).status_code == 204


def test_forgot_password_requests_are_rate_limited_per_client(env, monkeypatch):
    monkeypatch.setattr(env.service.by_reset_client, "max_calls", 2)
    assert forgot(env, "a@example.com").status_code == 204
    assert forgot(env, "b@example.com").status_code == 204
    blocked = forgot(env, "c@example.com")
    assert blocked.status_code == 429 and blocked.json()["code"] == "too_many_attempts" and int(blocked.headers["Retry-After"]) >= 1


def test_email_uniqueness_is_enforced_case_insensitively(env):
    register(env, "first-owner", email="shared@example.com")
    env.cookies.clear()
    taken = register(env, "second-owner", email="SHARED@EXAMPLE.COM")
    assert taken.status_code == 409 and taken.json()["code"] == "email_taken"


@pytest.mark.parametrize("bad_email", ["not-an-email", "missing-domain@", "@no-local-part.com", "spaces in@example.com"])
def test_register_rejects_malformed_email(env, bad_email):
    r = register(env, "bad-email-user", email=bad_email)
    assert r.status_code == 422 and r.json()["code"] == "invalid_email"


def test_users_can_set_and_clear_their_own_email_after_registering(env):
    register(env, "profile-user")
    r = env.put("/api/users/profile-user/email", json={"email": "profile-user@example.com"})
    assert r.status_code == 200 and r.json()["email"] == "profile-user@example.com"
    forgot(env, "profile-user@example.com")
    assert len(env.email.sent) == 1
    cleared = env.put("/api/users/profile-user/email", json={"email": None})
    assert cleared.status_code == 200 and cleared.json()["email"] is None


def test_setting_a_malformed_or_already_taken_email_via_the_profile_endpoint_fails(env):
    register(env, "owner-user", email="owner@example.com")
    env.cookies.clear()
    register(env, "other-user")
    bad = env.put("/api/users/other-user/email", json={"email": "not-an-email"})
    assert bad.status_code == 422 and bad.json()["code"] == "invalid_email"
    taken = env.put("/api/users/other-user/email", json={"email": "owner@example.com"})
    assert taken.status_code == 409 and taken.json()["code"] == "email_taken"


def test_reset_password_handles_the_token_being_consumed_between_peek_and_consume(env, monkeypatch):
    """A rare concurrent-request race: the token was still valid when checked, but gone by the time it's consumed."""
    register(env, "race-user", email="race-user@example.com")
    forgot(env, "race-user@example.com")
    token = token_from(env.email.sent[0])
    monkeypatch.setattr(env.service.resets, "consume", lambda token_hash: None)
    assert reset(env, token, GOOD).status_code == 400


def test_email_send_failure_is_logged_and_never_surfaced_to_the_caller(env, caplog):
    register(env, "flaky-provider", email="flaky@example.com")
    from src.core.errors import EmailError
    env.email.error = EmailError("SES is down")
    r = forgot(env, "flaky@example.com")
    assert r.status_code == 204  # the caller sees success either way
    assert "failed to send" in caplog.text.lower()


# ------------------------------------------------------------------------- protection and ownership
def test_every_route_except_the_public_ones_requires_sign_in(env):
    """Guards against a future route being added without protection."""
    spec = env.get("/openapi.json").json()["paths"]
    checked = 0
    for path, methods in spec.items():
        for method in methods:
            url = path.replace("{user_id}", "learner-105").replace("{conversation_id}", "c1").replace("{symbol}", "AAPL").replace("{goal_id}", "1")
            if url in PUBLIC_PATHS or (url == "/api/glossary" and method == "get"):
                continue
            r = env.request(method.upper(), url, json={} if method != "get" else None)
            assert r.status_code == 401 and r.json()["code"] == "not_authenticated", f"{method.upper()} {path} -> {r.status_code}"
            checked += 1
    assert checked >= 25


def test_public_endpoints_stay_open(env):
    assert env.get("/api/health").status_code == 200 and env.get("/api/auth/config").status_code == 200
    assert env.get("/api/glossary").status_code == 200 and env.get("/api/auth/me").status_code == 401


def test_users_can_only_reach_their_own_data(env):
    register(env, "alice-user")
    assert env.get("/api/portfolio/alice-user").status_code == 200
    for method, url, body in [("GET", "/api/portfolio/learner-105", None), ("PUT", "/api/portfolio/learner-105", {"holdings": [], "cash": 0}),
                              ("GET", "/api/portfolio/learner-105/analysis", None), ("GET", "/api/portfolio/learner-105/comparison", None),
                              ("GET", "/api/goals/learner-105", None), ("POST", "/api/goals/learner-105", {"name": "x", "target_amount": 1, "years": 1}),
                              ("GET", "/api/conversations/learner-105/c1", None), ("DELETE", "/api/conversations/learner-105/c1", None),
                              ("GET", "/api/users/learner-105", None), ("PUT", "/api/users/learner-105/profile", {"horizon_years": 5}),
                              ("PUT", "/api/users/learner-105/email", {"email": "hijack@example.com"})]:
        r = env.request(method, url, json=body)
        assert r.status_code == 403 and r.json()["code"] == "forbidden", f"{method} {url} -> {r.status_code}"
    assert env.get("/api/portfolio/ALICE-USER").status_code == 200  # the caller's own id in any letter case


def test_chat_as_someone_else_is_forbidden(env):
    register(env, "alice-user")
    r = env.post("/api/chat", json={"user_id": "learner-105", "conversation_id": "c", "question": "What is an ETF?"})
    assert r.status_code == 403 and r.json()["code"] == "forbidden"


def test_goal_projection_uses_the_signed_in_users_profile_not_a_body_id(env):
    login(env, "learner-101", settings.demo_password)  # aggressive
    body = {"name": "x", "target_amount": 50000, "years": 8, "current_savings": 5000, "monthly_contribution": 300, "user_id": "learner-102"}  # conservative
    assert env.post("/api/goals/project", json=body).json()["goal"]["risk_tolerance"] == "aggressive"


def test_upload_endpoints_are_admin_only(env, monkeypatch):
    register(env, "plain-user")
    kb = ("kb.json", "[]", "application/json")
    assert env.post("/api/ingest/file", files={"file": kb}).json()["code"] == "admin_only"
    assert env.post("/api/glossary/initialize", files={"file": kb}).status_code == 403
    assert env.get("/api/auth/me").json()["is_admin"] is False
    monkeypatch.setattr(auth_mod, "admin_ids", lambda: frozenset({"plain-user"}))
    assert env.get("/api/auth/me").json()["is_admin"] is True
    assert env.post("/api/ingest/file", files={"file": ("kb.txt", "x", "text/plain")}).status_code == 400  # passed the admin gate


def test_state_changing_requests_need_the_app_header(env):
    bare = TestClient(main.app)  # no X-Clarity-Client header, like a cross-site form post
    for url, body in [("/api/auth/login", {"user_id": "learner-105", "password": settings.demo_password}), ("/api/auth/register", {"user_id": "csrf-user", "password": GOOD})]:
        r = bare.post(url, json=body)
        assert r.status_code == 403 and r.json()["code"] == "csrf_blocked"
    register(env, "csrf-victim")
    cookie = env.cookies.get("clarity_session")
    r = TestClient(main.app, cookies={"clarity_session": cookie}).put("/api/portfolio/csrf-victim", json={"holdings": [], "cash": 5})
    assert r.status_code == 403 and r.json()["code"] == "csrf_blocked"
    assert TestClient(main.app, cookies={"clarity_session": cookie}).get("/api/portfolio/csrf-victim").status_code == 200  # reads are fine


# --------------------------------------------------------------------------------------------- demo
def test_config_and_demo_switch(env, monkeypatch):
    cfg = env.get("/api/auth/config").json()
    assert cfg["demo_login"] is True and cfg["demo_password"] == settings.demo_password and cfg["demo_users"][0] == "learner-101" and cfg["min_password_length"] == 8
    monkeypatch.setattr(auth_mod, "settings", type("S", (), {"demo_login_enabled": False, "demo_password": "x", "admin_user_ids": ""})())
    with env.db.connect() as conn:
        conn.execute("UPDATE users SET password_hash = NULL WHERE user_id = 'learner-101'")
    assert env.service.ensure_demo_passwords() == 0
    monkeypatch.setattr("src.web_app.api.routes_auth.settings", type("S", (), {"demo_login_enabled": False, "demo_password": "x"})())
    off = env.get("/api/auth/config").json()
    assert off["demo_login"] is False and off["demo_users"] == [] and off["demo_password"] is None


def test_principal_ownership_rules():
    assert Principal("Alice").owns("alice") and not Principal("alice").owns("bob") and Principal("x", any_user=True).owns("anyone")
    assert not Principal("alice").owns("")
