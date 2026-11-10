

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
