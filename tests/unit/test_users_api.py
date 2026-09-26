"""User profile endpoints (sign-in itself is covered in test_auth.py)."""
import pytest
from fastapi.testclient import TestClient

from src.data.db import Database
from src.data.seed import seed
from src.web_app.api.main import app, get_db


@pytest.fixture
def client(tmp_path):
    db = Database(tmp_path / "api.db")
    db.init_schema()
    seed(db)
    app.dependency_overrides[get_db] = lambda: db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_demo_list_has_ten_synthetic_users_and_excludes_registered_ones(client):
    ids = [u["user_id"] for u in client.get("/api/users").json()]
    assert len(ids) == 10 and ids[0] == "learner-101"
    from src.data.repositories import UserRepository
    UserRepository(app.dependency_overrides[get_db]()).ensure("some-real-person")
    assert "some-real-person" not in [u["user_id"] for u in client.get("/api/users").json()]


def test_get_user_and_unknown_user(client):
    body = client.get("/api/users/learner-105").json()
    assert body["experience_level"] == "advanced" and body["user_id"] == "learner-105"
    assert client.get("/api/users/nobody-here").status_code == 404
