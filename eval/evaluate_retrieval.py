"""Evaluate article retrieval using a Langfuse dataset experiment."""

import argparse


from langfuse import get_client
from metrics import article_hit_evaluator, article_recall_evaluator
from rag_llmops.config import load_config
from rag_llmops.retrieval import resolve_qdrant_location
from rag_llmops.embeddings import load_embedding_model
from rag_llmops.retrieval import (
    load_retriever,
    retrieve,
    load_index,
    route_chapters,
    build_retriever,
    get_client as get_qdrant_client,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate article retrieval with Langfuse."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Golden_Dataset",
        help="Langfuse dataset name (default: Golden_Dataset).",
    )
    parser.add_argument(
        "--route",
        action="store_true",
        help="Enable automatic chapter routing before retrieval.",
    )
    parser.add_argument(
        "--top-chapters",
        type=int,
        default=2,
        help="Number of chapters selected by the routing stage.",
    )
    parser.add_argument(
        "--route-pool",
        type=int,
        default=30,
        help="Number of global dense results used for chapter routing.",
    )

    args = parser.parse_args()

    if args.top_chapters < 1 or args.route_pool < 1:
        parser.error("--top-chapters and --route-pool must be positive.")

    langfuse = get_client()

    if not langfuse.auth_check():
        raise RuntimeError("Langfuse authentication failed.")

    config = load_config()
    retrieval_config = config["retrieval"]
    qdrant_location = resolve_qdrant_location(config["index"])
    collection_name = config["index"]["collection_name"]
    mode = retrieval_config["mode"]
    top_k = retrieval_config["top_k"]
    candidate_k = retrieval_config["candidate_k"]

    qdrant_client = None

    try:
        if args.route:
            # Load the index once; route each question separately.
            index = load_index(
                qdrant_location=qdrant_location,
                collection_name=collection_name,
                embed_model=load_embedding_model(config["embedding"]),
            )
            qdrant_client = get_qdrant_client(qdrant_location)
            retriever = None

            experiment_name = (
                f"Retrieval routed - top {args.top_chapters} chapters"
            )
        else:
            # Original baseline: global retrieval, without routing.
            index = None
            retriever = load_retriever(
                qdrant_location=qdrant_location,
                collection_name=collection_name,
                embed_model=load_embedding_model(config["embedding"]),
                k=top_k,
                mode=mode,
                candidate_k=candidate_k,
            )
            experiment_name = "Retrieval baseline"

        # Load the dataset selected from the command line.
        dataset = langfuse.get_dataset(args.dataset)

        print(f"Dataset: {dataset.name} | Items: {len(dataset.items)}")
        print(f"Mode: {mode}")
        print(
            f"Chapter routing: "
            f"{'enabled' if args.route else 'disabled'}"
        )

        def retrieval_task(*, item, **kwargs):
            question = item.input["question"]

            if args.route:
                routed = route_chapters(
                    index=index,
                    query=question,
                    top_n=args.top_chapters,
                    pool=args.route_pool,
                )
                chapters = [chapter for chapter, _ in routed]

                if not chapters:
                    # No chapter could be identified; use global retrieval.
                    question_retriever = build_retriever(
                        index=index,
                        qdrant_location=qdrant_location,
                        collection_name=collection_name,
                        k=top_k,
                        mode=mode,
                        candidate_k=candidate_k,
                    )
                else:
                    question_retriever = build_retriever(
                        index=index,
                        qdrant_location=qdrant_location,
                        collection_name=collection_name,
                        k=top_k,
                        mode=mode,
                        candidate_k=candidate_k,
                        chapter=chapters,
                    )

                chunks = retrieve(question_retriever, question)
            else:
                chunks = retrieve(retriever, question)

            articles = sorted(
                {
                    str(chunk["article"])
                    for chunk in chunks
                    if chunk.get("article") is not None
                }
            )

            return {
                "question": question,
                "retrieved_articles": articles,
                "chunks": chunks,
            }

        result = dataset.run_experiment(
            name=experiment_name,
            description=(
                "Evaluate whether expected articles appear in retrieved "
                "chunks. Chapter routing is "
                f"{'enabled' if args.route else 'disabled'}. "
                f"Dataset: {args.dataset}."
            ),
            task=retrieval_task,
            evaluators=[article_hit_evaluator, article_recall_evaluator],
        )

        print(result.format())

        if getattr(result, "experiment_url", None):
            print("Experiment:", result.experiment_url)

    finally:
        if qdrant_client is not None:
            qdrant_client.close()
            get_qdrant_client.cache_clear()

        langfuse.flush()


if __name__ == "__main__":
    main()