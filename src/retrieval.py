from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


def load_retriever(persist_dir: str = "storage", model_name: str = "sentence-transformers/all-MiniLM-L6-v2", k: int = 5):
    embed_model = HuggingFaceEmbedding(model_name=model_name)

    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
    index = load_index_from_storage(storage_context, embed_model=embed_model)

    return index.as_retriever(similarity_top_k=k)


def retrieve(retriever, query: str) -> list[dict]:
    results = retriever.retrieve(query)
    return [
        {
            "text": r.text,
            "page": r.node.metadata.get("page_label"),
            "score": r.score,
        }
        for r in results
    ]


if __name__ == "__main__":
    from config import load_config

    config = load_config()
    retriever = load_retriever(
        persist_dir=config["index"]["persist_dir"],
        model_name=config["embedding"]["model_name"],
        k=config["retrieval"]["top_k"],
    )

    query = "What are the prohibited AI practices?"
    results = retrieve(retriever, query)

    print(f"Query: {query}\n")
    for r in results:
        print(f"[{r['score']:.3f}] page {r['page']} | {r['text'][:100]!r}")
