"""Embedding backend selection. No model download or server needed."""

import pytest
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference

from rag_llmops.embeddings import load_embedding_model

BGE = "BAAI/bge-base-en-v1.5"
# Same instruction the in-process (hf) backend adds for BGE models.
BGE_QUERY_INSTRUCTION = "Represent this question for searching relevant passages:"
TEI_CONFIG = {"backend": "tei", "model_name": BGE, "base_url": "http://tei:80"}


def test_tei_backend_points_at_the_server():
    model = load_embedding_model(TEI_CONFIG)
    assert isinstance(model, TextEmbeddingsInference)
    assert model.base_url == "http://tei:80"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        load_embedding_model({"backend": "nope", "model_name": BGE})


def test_tei_adds_bge_instruction_to_queries_only(monkeypatch):
    """BGE expects an instruction on queries but not on documents. If the
    client sent queries without it, retrieval would degrade silently."""
    sent = []
    monkeypatch.setattr(
        TextEmbeddingsInference,
        "_call_api",
        lambda self, texts: sent.extend(texts) or [[0.0]] * len(texts),
    )
    model = load_embedding_model(TEI_CONFIG)

    model.get_query_embedding("Which practices are prohibited?")
    model.get_text_embedding("Article 5 prohibits certain AI practices.")

    query_sent, text_sent = sent
    assert query_sent.startswith(BGE_QUERY_INSTRUCTION)
    assert query_sent.endswith("Which practices are prohibited?")
    assert text_sent == "Article 5 prohibits certain AI practices."
