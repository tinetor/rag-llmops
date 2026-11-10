"""Embedding model selection: in-process (hf) or a TEI server (tei).

Ingestion and queries both call load_embedding_model() with the same config,
so the index and the questions are always embedded the same way.
"""

from llama_index.core.base.embeddings.base import BaseEmbedding


def load_embedding_model(embedding_config: dict) -> BaseEmbedding:
    """Return the embedding model configured under the `embedding` key."""
    backend = embedding_config.get("backend", "hf")
    model_name = embedding_config["model_name"]

    if backend == "hf":
        # Imported lazily: needs torch, which the tei backend does not.
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=model_name)

    if backend == "tei":
        from llama_index.embeddings.text_embeddings_inference import (
            TextEmbeddingsInference,
        )

        # model_name must be the Hugging Face model id the TEI server runs:
        # the client uses it to add the BGE query instruction to questions.
        return TextEmbeddingsInference(
            model_name=model_name,
            base_url=embedding_config["base_url"],
            timeout=embedding_config.get("timeout", 60),
        )

    raise ValueError(f"Unknown embedding backend: {backend!r}")
