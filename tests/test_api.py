"""API contract tests. Loading and answering are replaced by fakes, so these
tests need no models, index or GPU."""

import pytest
from fastapi.testclient import TestClient

from rag_llmops import api


FAIL = {"enabled": False}
CLOSED = []


def fake_answer(components, question):
    if FAIL["enabled"]:
        raise RuntimeError("internal details")
    return {
        "answer": f"Answer to: {question}",
        "escaped": False,
        "sources": [
            {
                "text": "Article 5 prohibits certain AI practices.",
                "article": 5,
                "chapter": "II",
                "chapter_title": "PROHIBITED AI PRACTICES",
                "score": 0.033,
            }
        ],
    }


@pytest.fixture
def client(monkeypatch):
    FAIL["enabled"] = False
    CLOSED.clear()
    monkeypatch.setattr(api, "load_components", lambda: "components")
    monkeypatch.setattr(api, "answer", fake_answer)
    monkeypatch.setattr(api, "close_components", CLOSED.append)
    with TestClient(api.app) as test_client:  # runs the lifespan
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_and_sources(client):
    response = client.post("/query", json={"question": "What is prohibited?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Answer to: What is prohibited?"
    assert body["escaped"] is False
    assert body["sources"][0]["article"] == 5


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_error_does_not_leak_details(client):
    FAIL["enabled"] = True
    response = client.post("/query", json={"question": "anything"})
    assert response.status_code == 500
    assert "internal details" not in response.text


def test_components_are_closed_on_shutdown(monkeypatch):
    closed = []
    monkeypatch.setattr(api, "load_components", lambda: "components")
    monkeypatch.setattr(api, "close_components", closed.append)
    with TestClient(api.app):
        assert closed == []
    assert closed == ["components"]
