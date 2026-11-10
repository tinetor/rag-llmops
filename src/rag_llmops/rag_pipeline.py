"""Orchestrate retrieval and answer generation."""

from langfuse import get_client as get_langfuse_client

from rag_llmops.config import load_config
from rag_llmops.generation import generate_answer
from rag_llmops.llm import load_llm
from rag_llmops.retrieval import get_client as get_qdrant_client
from rag_llmops.retrieval import load_index, load_retriever


def load_components(config_path="configs/config.yaml") -> dict:
    """Load config, index or retriever, and the LLM. Slow: call once.

    Returns a dict with the keys: config, index, retriever, model, tokenizer.
    """
    config = load_config(config_path)
    retrieval = config["retrieval"]
    persist_dir = config["index"]["persist_dir"] + "_qdrant"

    index = None
    retriever = None
    if retrieval.get("use_router", False):
        # The router builds a filtered retriever per question,
        # so only the index is loaded up front.
        index = load_index(
            persist_dir=persist_dir,
            collection_name=config["index"]["collection_name"],
            model_name=config["embedding"]["model_name"],
        )
    else:
        retriever = load_retriever(
            persist_dir=persist_dir,
            collection_name=config["index"]["collection_name"],
            model_name=config["embedding"]["model_name"],
            k=retrieval["top_k"],
            mode=retrieval["mode"],
            candidate_k=retrieval["candidate_k"],
        )

    model, tokenizer = load_llm(config)
    return {
        "config": config,
        "index": index,
        "retriever": retriever,
        "model": model,
        "tokenizer": tokenizer,
    }


def answer(components: dict, question: str) -> dict:
    """Answer one question with already-loaded components. Fast: call often."""
    config = components["config"]
    retrieval = config["retrieval"]
    generation = config["generation"]

    result = generate_answer(
        question=question,
        retriever=components["retriever"],
        model=components["model"],
        tokenizer=components["tokenizer"],
        max_new_tokens=generation["max_new_tokens"],
        top_k=retrieval["top_k"],
        backend=generation.get("backend", "hf"),
        model_name=generation["model_name"],
        use_router=retrieval.get("use_router", False),
        index=components["index"],
        persist_dir=config["index"]["persist_dir"] + "_qdrant",
        collection_name=config["index"]["collection_name"],
        embedding_model_name=config["embedding"]["model_name"],
        top_chapters=retrieval.get("top_chapters", 2),
        route_pool=retrieval.get("route_pool", 30),
        retrieval_mode=retrieval["mode"],
        candidate_k=retrieval["candidate_k"],
    )

    # Flush pending Langfuse traces before returning.
    get_langfuse_client().flush()

    return result


def close_components(components: dict) -> None:
    """Release shared clients on shutdown. Safe to call once at exit."""
    config = components["config"]

    # OpenAI client used for the vLLM (http) backend keeps a connection pool.
    if config["generation"].get("backend", "hf") == "http":
        components["model"].close()

    # Qdrant client is cached per storage path; close it and drop the cache.
    persist_dir = config["index"]["persist_dir"] + "_qdrant"
    get_qdrant_client(persist_dir).close()
    get_qdrant_client.cache_clear()

    # Send any pending traces and stop Langfuse background workers.
    get_langfuse_client().shutdown()


def run_rag(question: str, config_path="configs/config.yaml") -> dict:
    """Load everything and answer a single question (CLI use)."""
    components = load_components(config_path)
    try:
        return answer(components, question)
    finally:
        close_components(components)
