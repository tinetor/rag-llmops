"""Startup guard for the index. Uses local Qdrant storage; no services needed."""

import pytest

from rag_llmops.retrieval import get_client, load_index


def test_load_index_fails_fast_without_collection(tmp_path):
    """Regression: an index built before ingestion guessed the wrong vector
    format and every query failed with 'vector name text-dense not found'."""
    location = str(tmp_path / "qdrant")
    try:
        with pytest.raises(RuntimeError, match="Run the ingestion job first"):
            load_index(
                qdrant_location=location,
                collection_name="missing",
                embed_model=None,
            )
    finally:
        get_client(location).close()
        get_client.cache_clear()
