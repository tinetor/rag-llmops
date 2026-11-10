"""Build the Qdrant collection from document chunks."""

from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def build_index(
    nodes,
    embed_model,
    persist_dir: str = "./storage_qdrant",
    collection_name: str = "eu_ai_act",
):
    """Embed the nodes and write them to a fresh Qdrant collection."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=persist_dir)

    # Ingestion is a full rebuild: drop the existing collection so that
    # re-running ingestion never appends a second copy of every chunk.
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    return VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
