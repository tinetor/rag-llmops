from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


def load_retriever(
    persist_dir: str,
    model_name: str,
    k: int,
    mode: str,
    candidate_k: int,
):
    if mode not in ("dense", "hybrid"):
        raise ValueError(
            f"Unknown retrieval mode: {mode!r}. Use 'dense' or 'hybrid'."
        )

    embed_model = HuggingFaceEmbedding(model_name=model_name)

    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=persist_dir,
    )

    index = load_index_from_storage(
        storage_context,
        embed_model=embed_model,
    )

    if mode == "dense":
        return index.as_retriever(similarity_top_k=k)

    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.retrievers.bm25 import BM25Retriever

    pool = max(candidate_k, k)
    vector_retriever = index.as_retriever(similarity_top_k=pool)

    nodes = [
        node
        for node in index.docstore.docs.values()
        if isinstance(node, TextNode)
    ]

    if not nodes:
        raise ValueError(
            "No TextNodes found in the persisted document store."
        )

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=pool,
    )

    Settings.llm = None

    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )


def retrieve(retriever, query: str) -> list[dict]:
    results = retriever.retrieve(query)

    return [
        {
            "text": result.text,
            "article": result.node.metadata.get("article"),
            "chapter": result.node.metadata.get("chapter"),
            "chapter_title": result.node.metadata.get("chapter_title"),
            "score": result.score,
        }
        for result in results
    ]


if __name__ == "__main__":
    from config import load_config

    config = load_config()

    retriever = load_retriever(
        persist_dir=config["index"]["persist_dir"],
        model_name=config["embedding"]["model_name"],
        k=config["retrieval"]["top_k"],
        mode=config["retrieval"]["mode"],
        candidate_k=config["retrieval"]["candidate_k"],
    )

    query = "What are the prohibited AI practices?"
    results = retrieve(retriever, query)

    print(f"Query: {query}\n")

    for result in results:
        score = (
            f"{result['score']:.3f}"
            if result["score"] is not None
            else "N/A"
        )

        article = result["article"]
        article_label = (
            f"Article {article}"
            if article is not None
            else "Article unknown"
        )

        chapter = result["chapter"]
        chapter_label = (
            f"Chapter {chapter} - {result['chapter_title']}"
            if chapter is not None
            else "Chapter unknown"
        )

        print(
            f"[{score}] {chapter_label} | {article_label} | "
            f"{result['text'][:120]!r}"
        )
