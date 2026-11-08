
from pathlib import Path

from qdrant_client import QdrantClient
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore


def build_index(
    nodes,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_dir: str = "./storage_qdrant",
    collection_name: str = "eu_ai_act",
):
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


if __name__ == "__main__":
    from chunking import chunking
    from config import load_config
    from load_pdf import load_pdf

    config = load_config()
    docs = load_pdf(config["data"]["pdf_path"])

    nodes = chunking(
        docs,
        max_chars=config["chunking"]["max_chars"],
        overlap_ratio=config["chunking"]["overlap_ratio"],
        strategy=config["chunking"]["strategy"],
    )

    index = build_index(
        nodes,
        model_name=config["embedding"]["model_name"],
        persist_dir=config["index"]["persist_dir"] + "_qdrant",
    )

    print(f"Qdrant index built: {len(nodes)} nodes")
