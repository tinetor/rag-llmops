"""Tests for indexing with Qdrant (downloads the embedding model)."""

import pytest
from llama_index.core.schema import TextNode

from rag_llmops.config import load_config
from rag_llmops.embeddings import load_embedding_model
from rag_llmops.indexing import build_index


SAMPLE_NODES = [
    TextNode(text="The quick brown fox jumps over the lazy dog."),
    TextNode(text="Article 5 prohibits certain AI practices."),
    TextNode(text="This regulation applies to providers and deployers."),
]


@pytest.fixture(scope="module")
def embed_model():
    # Loaded once for the whole module: model loading is the slow part.
    return load_embedding_model(load_config()["embedding"])


@pytest.fixture
def index(tmp_path, embed_model):
    return build_index(
        nodes=SAMPLE_NODES,
        embed_model=embed_model,
        qdrant_location=str(tmp_path / "qdrant"),
        collection_name="test_collection",
    )


def test_no_nodes_lost(index):
    stored_count = index.vector_store.client.count(
        collection_name="test_collection"
    ).count

    assert stored_count == len(SAMPLE_NODES)


def test_embedding_dimension_matches_model(index, embed_model):
    collection = index.vector_store.client.get_collection(
        collection_name="test_collection"
    )

    stored_dimension = collection.config.params.vectors.size
    expected_dimension = len(
        embed_model.get_text_embedding("Embedding dimension verification.")
    )

    assert stored_dimension == expected_dimension


def test_reingestion_replaces_collection(tmp_path, embed_model):
    """Running ingestion twice must not duplicate chunks."""
    kwargs = dict(
        nodes=SAMPLE_NODES,
        embed_model=embed_model,
        qdrant_location=str(tmp_path / "qdrant"),
        collection_name="test_collection",
    )

    # Close the first client: local Qdrant allows one client per folder.
    build_index(**kwargs).vector_store.client.close()
    index = build_index(**kwargs)

    stored_count = index.vector_store.client.count(
        collection_name="test_collection"
    ).count
    assert stored_count == len(SAMPLE_NODES)
