"""Orchestrate the document ingestion pipeline."""

from config import load_config
from load_pdf import load_pdf
from chunking import chunking
from indexing import build_index


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
        model_name=config["embedding"]["model_name"],
        persist_dir=config["index"]["persist_dir"] + "_qdrant",
        collection_name=config["index"]["collection_name"],
    )

    return index, nodes
