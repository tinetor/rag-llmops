
from pathlib import Path
from rag_llmops.config import load_config
from qdrant_client import QdrantClient
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore


def build_index(
    nodes,
    model_name: str | None = None,
    persist_dir: str = "./storage_qdrant",
    collection_name: str = "eu_ai_act",
):
    if model_name is None:
        config = load_config()
        model_name = config["embedding"]["model_name"]

    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    embed_model = HuggingFaceEmbedding(model_name=model_name)

    client = QdrantClient(path=persist_dir)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    return index
