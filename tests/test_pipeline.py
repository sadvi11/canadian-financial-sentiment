"""Unit tests for Canadian Financial Sentiment API"""
import json
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert data["service"] == "canadian-financial-sentiment"


def test_predict_requires_headline(client):
    response = client.post("/predict",
                           data=json.dumps({}),
                           content_type="application/json")
    assert response.status_code == 400


def test_predict_empty_headline(client):
    response = client.post("/predict",
                           data=json.dumps({"headline": ""}),
                           content_type="application/json")
    assert response.status_code == 400


def test_batch_requires_headlines(client):
    response = client.post("/batch",
                           data=json.dumps({}),
                           content_type="application/json")
    assert response.status_code == 400


def test_health_has_required_fields(client):
    response = client.get("/health")
    data = json.loads(response.data)
    assert "endpoint" in data
    assert "algorithm" in data
    assert "labels" in data
    assert "use_case" in data
