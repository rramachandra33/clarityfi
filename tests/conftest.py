"""Test isolation: never let tests touch the developer's real databases or indexes.

These env vars must be set before `src.core.config` is first imported, so this runs at conftest load.
"""
import atexit
import os
import shutil
import tempfile

_tmp = tempfile.mkdtemp(prefix="clarityfi-tests-")
os.environ["APP_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ["QDRANT_PATH"] = os.path.join(_tmp, "qdrant")
os.environ["FAISS_PATH"] = os.path.join(_tmp, "faiss")
os.environ["GLOSSARY_DB_PATH"] = os.path.join(_tmp, "glossary.db")
atexit.register(shutil.rmtree, _tmp, ignore_errors=True)


import sys  # noqa: E402

import pytest  # noqa: E402

from src.core import auth as _auth  # noqa: E402

_auth.SCRYPT_N = 2 ** 10  # fast hashing in tests; stored hashes carry their own parameters, so verification is unaffected


@pytest.fixture(autouse=True)
def _signed_in_unless_testing_auth(request):
    """Most tests are about features, not sign-in: act as a signed-in user who may access anyone's data.

    Tests that exercise real authentication carry `@pytest.mark.real_auth` and get the genuine dependency."""
    from fastapi import Request

    from src.web_app.api import main
    from src.web_app.auth_service import Principal
    from src.web_app.security import require_login

    if "real_auth" not in request.keywords:
        def allow(http: Request) -> None:
            http.state.principal = Principal(user_id="test-user", is_admin=True, any_user=True)

        main.app.dependency_overrides[require_login] = allow
    yield
    main.app.dependency_overrides.pop(require_login, None)


@pytest.fixture(autouse=True)
def _fresh_rate_limiters():
    """Rate limits are per process; give every test a clean slate so one test cannot starve the next."""
    main = sys.modules.get("src.web_app.api.main")
    if main is not None and hasattr(main, "api_services"):
        from src.web_app.api.services import default_limiters

        main.api_services.limiters = default_limiters()
    yield
