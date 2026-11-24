"""UI logic tests. The API is replaced by a fake; no server is started."""

import httpx
import pytest

import app


class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://api/query")
            raise httpx.HTTPStatusError(
                "error", request=request, response=httpx.Response(self.status_code)
            )


ANSWER = {
    "answer": "Certain practices are prohibited [Article 5].",
    "escaped": False,
    "sources": [
        {"text": "Article 5 " + "x" * 500, "article": 5, "chapter": "II", "score": 0.0331},
    ],
}


def test_answer_and_sources_are_shown(monkeypatch):
    monkeypatch.setattr(app.httpx, "post", lambda *a, **k: FakeResponse(200, ANSWER))
    cleared, history, rows = app.ask("What is prohibited?", [])

    assert cleared == ""
    assert history[0] == {"role": "user", "content": "What is prohibited?"}
    assert history[1]["content"] == ANSWER["answer"]
    article, chapter, score, passage = rows[0]
    assert (article, chapter, score) == (5, "II", "0.033")
    assert len(passage) <= app.PASSAGE_PREVIEW + 3


def test_escaped_answer_is_flagged(monkeypatch):
    body = dict(ANSWER, escaped=True)
    monkeypatch.setattr(app.httpx, "post", lambda *a, **k: FakeResponse(200, body))
    _, history, _ = app.ask("Off-topic question", [])
    assert "does not cover" in history[-1]["content"]


@pytest.mark.parametrize("failure, expected", [
    (lambda *a, **k: FakeResponse(500), "error (500)"),
    (lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("down")), "could not be reached"),
])
def test_api_failures_are_shown_in_the_chat(monkeypatch, failure, expected):
    monkeypatch.setattr(app.httpx, "post", failure)
    _, history, rows = app.ask("Anything", [])
    assert expected in history[-1]["content"]
    assert rows == []


def test_empty_question_does_nothing(monkeypatch):
    monkeypatch.setattr(app.httpx, "post", lambda *a, **k: pytest.fail("API called"))
    _, history, _ = app.ask("   ", [])
    assert history == []


def test_ui_builds():
    assert app.build_ui() is not None
