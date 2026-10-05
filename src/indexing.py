import faiss
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


def build_index(nodes, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embed_model = HuggingFaceEmbedding(model_name=model_name)


    embed_dim = len(embed_model.get_text_embedding("probe"))

    faiss_index = faiss.IndexFlatIP(embed_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    return index


if __name__ == "__main__":
    from config import load_config
    from load_pdf import load_pdf
    from chunking import chunking

    config = load_config()
    docs = load_pdf(config["data"]["pdf_path"])
    nodes = chunking(docs, max_chars=config["chunking"]["max_chars"])

    index = build_index(nodes)
    index.storage_context.persist(persist_dir=config["index"]["persist_dir"])
    print(f"Index built and persisted: {len(nodes)} nodes")
