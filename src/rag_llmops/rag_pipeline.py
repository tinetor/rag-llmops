"""Orchestrate retrieval and answer generation."""

from langfuse import get_client

from rag_llmops.config import load_config
from rag_llmops.llm import load_llm
from rag_llmops.retrieval import load_index, load_retriever
from rag_llmops.generation import generate_answer


def run_rag(question: str, config_path="configs/config.yaml") -> dict:
    """Load RAG components and generate an answer for a question."""
    config = load_config(config_path)
    retrieval_config = config["retrieval"]

    persist_dir = config["index"]["persist_dir"] + "_qdrant"
    collection_name = config["index"]["collection_name"]
    embedding_model_name = config["embedding"]["model_name"]

    use_router = retrieval_config.get("use_router", False)
    top_k = retrieval_config["top_k"]
    candidate_k = retrieval_config["candidate_k"]
    mode = retrieval_config["mode"]
    top_chapters = retrieval_config.get("top_chapters", 2)
    route_pool = retrieval_config.get("route_pool", 30)

    if use_router:
        index = load_index(
            persist_dir=persist_dir,
            collection_name=collection_name,
            model_name=embedding_model_name,
        )
        retriever = None
    else:
        index = None
        retriever = load_retriever(
            persist_dir=persist_dir,
            collection_name=collection_name,
            model_name=embedding_model_name,
            k=top_k,
            mode=mode,
            candidate_k=candidate_k,
        )

    model, tokenizer = load_llm(config)

    result = generate_answer(
        question=question,
        retriever=retriever,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config["generation"]["max_new_tokens"],
        top_k=top_k,
        backend=config["generation"].get("backend", "hf"),
        model_name=config["generation"]["model_name"],
        use_router=use_router,
        index=index,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        top_chapters=top_chapters,
        route_pool=route_pool,
        retrieval_mode=mode,
        candidate_k=candidate_k,
    )

    # Flush pending Langfuse traces before returning.
    get_client().flush()

    return result
