"""Pipeline wiring tests. Heavy components are replaced by fakes."""

from types import SimpleNamespace

from rag_llmops import rag_pipeline


def test_components_load_once_and_config_is_forwarded(monkeypatch):
    loads = {"llm": 0}
    calls = []
    flushes = []

    def fake_load_llm(config):
        loads["llm"] += 1
        return "model", "tokenizer"

    def fake_generate_answer(**kwargs):
        calls.append(kwargs)
        return {"answer": "ok", "escaped": False, "sources": []}

    monkeypatch.setattr(rag_pipeline, "load_index", lambda **kwargs: "index")
    monkeypatch.setattr(rag_pipeline, "load_retriever", lambda **kwargs: "retriever")
    monkeypatch.setattr(rag_pipeline, "load_llm", fake_load_llm)
    monkeypatch.setattr(rag_pipeline, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(
        rag_pipeline,
        "get_langfuse_client",
        lambda: SimpleNamespace(flush=lambda: flushes.append(True)),
    )

    components = rag_pipeline.load_components()
    rag_pipeline.answer(components, "first question")
    rag_pipeline.answer(components, "second question")

    # Components are loaded once, not once per question.
    assert loads["llm"] == 1
    assert len(calls) == 2
    # Retrieval settings from the config reach generation unchanged.
    retrieval = components["config"]["retrieval"]
    assert calls[0]["top_k"] == retrieval["top_k"]
    assert calls[0]["candidate_k"] == retrieval["candidate_k"]
    assert len(flushes) == 2


def test_close_components_releases_clients(monkeypatch):
    events = []
    qdrant = SimpleNamespace(close=lambda: events.append("qdrant"))
    fake_get_qdrant = lambda persist_dir: qdrant
    fake_get_qdrant.cache_clear = lambda: events.append("cache_clear")

    monkeypatch.setattr(rag_pipeline, "get_qdrant_client", fake_get_qdrant)
    monkeypatch.setattr(
        rag_pipeline,
        "get_langfuse_client",
        lambda: SimpleNamespace(shutdown=lambda: events.append("langfuse")),
    )

    components = {
        "config": {
            "generation": {"backend": "http"},
            "index": {"persist_dir": "./storage"},
        },
        "model": SimpleNamespace(close=lambda: events.append("llm")),
    }
    rag_pipeline.close_components(components)

    assert events == ["llm", "qdrant", "cache_clear", "langfuse"]
