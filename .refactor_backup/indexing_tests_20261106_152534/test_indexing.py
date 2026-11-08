"""Tests for indexing with Qdrant."""

import pytest
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import load_config
from indexing import build_index


SAMPLE_NODES = [
    TextNode(text="The quick brown fox jumps over the lazy dog."),
    TextNode(text="Article 5 prohibits certain AI practices."),
    TextNode(text="This regulation applies to providers and deployers."),
]


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def index(tmp_path, config):
    return build_index(
        nodes=SAMPLE_NODES,
        model_name=config["embedding"]["model_name"],
        persist_dir=str(tmp_path / "qdrant"),
        collection_name="test_collection",
    )


def test_no_nodes_lost(index):
    stored_count = index.vector_store.client.count(
        collection_name="test_collection"
    ).count

    assert stored_count == len(SAMPLE_NODES)


def test_embedding_dimension_matches_model(index, config):
    collection = index.vector_store.client.get_collection(
        collection_name="test_collection"
    )

    stored_dimension = collection.config.params.vectors.size

    embedding_model = HuggingFaceEmbedding(
        model_name=config["embedding"]["model_name"]
    )
    expected_dimension = len(
        embedding_model.get_text_embedding("Embedding dimension verification.")
    )

    assert stored_dimension == expected_dimension
