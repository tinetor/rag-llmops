from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


def load_retriever(persist_dir: str = "storage", model_name: str = "sentence-transformers/all-MiniLM-L6-v2", k: int = 5):
    embed_model = HuggingFaceEmbedding(model_name=model_name)

    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
    index = load_index_from_storage(storage_context, embed_model=embed_model)

    return index.as_retriever(similarity_top_k=k)


if __name__ == "__main__":
    retriever = load_retriever()

    query = "What are the prohibited AI practices?"
    results = retriever.retrieve(query)

    print(f"Query: {query}\n")
    for r in results:
        print(f"[{r.score:.3f}] page {r.node.metadata.get('page_label')} | {r.text[:100]!r}")
