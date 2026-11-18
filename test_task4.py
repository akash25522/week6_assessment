import pytest
from fastapi.testclient import TestClient
from task4_api import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_search_without_api_key(client):
    response = client.post(
        "/search",
        json={"query": "company working hours"}
    )
    assert response.status_code == 422


def test_search_invalid_api_key(client):
    response = client.post(
        "/search",
        headers={"X-API-Key": "wrong-key"},
        json={"query": "company working hours"}
    )
    assert response.status_code == 401


def test_search_valid_request(client):
    response = client.post(
        "/search",
        headers={"X-API-Key": "week6-secret-key"},
        json={"query": "company working hours", "top_k": 5}
    )
    assert response.status_code == 200
    assert "results" in response.json()


def test_search_invalid_top_k(client):
    response = client.post(
        "/search",
        headers={"X-API-Key": "week6-secret-key"},
        json={"query": "company working hours", "top_k": 25}
    )
    assert response.status_code == 422


def test_upload_invalid_file_type(client):
    response = client.post(
        "/documents",
        headers={"X-API-Key": "week6-secret-key"},
        files={"file": ("test.jpg", b"sample content", "image/jpeg")}
    )
    assert response.status_code == 400