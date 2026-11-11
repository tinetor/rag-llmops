"""Build the Qdrant collection from document chunks."""

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from rag_llmops.retrieval import create_qdrant_client


def build_index(
    nodes,
    embed_model,
    qdrant_location: str = "./storage_qdrant",
    collection_name: str = "eu_ai_act",
):
    """Embed the nodes and write them to a fresh Qdrant collection."""
    client = create_qdrant_client(qdrant_location)

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
