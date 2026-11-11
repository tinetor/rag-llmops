"""Orchestrate the document ingestion pipeline."""

from rag_llmops.config import load_config
from rag_llmops.retrieval import resolve_qdrant_location
from rag_llmops.embeddings import load_embedding_model
from rag_llmops.load_pdf import load_pdf
from rag_llmops.chunking import chunking
from rag_llmops.indexing import build_index


def run_ingestion(config_path="configs/config.yaml"):
    """Load documents, create chunks, and build the Qdrant index."""
    config = load_config(config_path)

    documents = load_pdf(config["data"]["pdf_path"])

    nodes = chunking(
        documents,
        max_chars=config["chunking"]["max_chars"],
        overlap_ratio=config["chunking"]["overlap_ratio"],
        strategy=config["chunking"]["strategy"],
    )

    index = build_index(
        nodes,
        embed_model=load_embedding_model(config["embedding"]),
        qdrant_location=resolve_qdrant_location(config["index"]),
        collection_name=config["index"]["collection_name"],
    )

    return index, nodes
