"""Configuration overlay and Qdrant location. No services needed."""

from qdrant_client import QdrantClient

from rag_llmops.config import load_config
from rag_llmops.retrieval import create_qdrant_client, resolve_qdrant_location


def test_without_overlay_the_defaults_are_used(monkeypatch):
    monkeypatch.delenv("RAG_CONFIG_OVERLAY", raising=False)
    config = load_config()
    assert config["embedding"]["backend"] == "hf"
    assert "url" not in config["index"]


def test_docker_overlay_changes_only_what_it_declares(monkeypatch):
    monkeypatch.setenv("RAG_CONFIG_OVERLAY", "configs/config.docker.yaml")
    config = load_config()

    assert config["embedding"]["backend"] == "tei"
    assert config["index"]["url"] == "http://qdrant:6333"
    assert config["generation"]["base_url"] == "http://llm:8000/v1"
    # Values not in the overlay keep their defaults.
    assert config["embedding"]["model_name"] == "BAAI/bge-base-en-v1.5"
    assert config["retrieval"]["top_k"] == 8


def test_qdrant_location_prefers_the_server_url():
    assert resolve_qdrant_location(
        {"url": "http://qdrant:6333", "persist_dir": "./storage"}
    ) == "http://qdrant:6333"
    assert resolve_qdrant_location({"persist_dir": "./storage"}) == "./storage_qdrant"


def test_local_location_opens_local_storage(tmp_path):
    client = create_qdrant_client(str(tmp_path / "qdrant"))
    try:
        assert isinstance(client, QdrantClient)
        assert client.get_collections().collections == []
    finally:
        client.close()
