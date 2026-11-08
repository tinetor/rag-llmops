"""Tests for indexing: checks that building the FAISS index via LlamaIndex
preserves every node and uses the correct embedding dimension."""

from llama_index.core.schema import TextNode

from indexing import build_index

SAMPLE_NODES = [
    TextNode(text="The quick brown fox jumps over the lazy dog."),
    TextNode(text="Article 5 prohibits certain AI practices."),
    TextNode(text="This regulation applies to providers and deployers."),
]


def test_no_nodes_lost():
    index = build_index(SAMPLE_NODES)
    assert index.vector_store.client.ntotal == len(SAMPLE_NODES)


def test_embedding_dimension_matches_model():
    index = build_index(SAMPLE_NODES)
    # all-MiniLM-L6-v2's known output size -- not recalculated here on purpose,
    # so the test catches indexing.py passing the wrong dimension to FAISS
    assert index.vector_store.client.d == 384
