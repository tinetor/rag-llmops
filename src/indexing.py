import faiss
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


def build_index(nodes, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embed_model = HuggingFaceEmbedding(model_name=model_name)

    # ask the model for its real dimension instead of hardcoding it
    embed_dim = len(embed_model.get_text_embedding("probe"))

    faiss_index = faiss.IndexFlatIP(embed_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
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


    index = build_index(nodes, model_name=config["embedding"]["model_name"])
    index.storage_context.persist(persist_dir=config["index"]["persist_dir"])
    print(f"Index built and persisted: {len(nodes)} nodes")
