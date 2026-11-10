"""Evaluate RAG generation using a Langfuse dataset experiment."""

import argparse


from langfuse import get_client
from metrics import article_hit_evaluator, article_recall_evaluator
from rag_llmops.config import load_config
from rag_llmops.embeddings import load_embedding_model
from rag_llmops.generation import generate_answer
from rag_llmops.llm import load_llm
from rag_llmops.retrieval import (
    load_retriever,
    retrieve,
    load_index,
    route_chapters,
    build_retriever,
    get_client as get_qdrant_client,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG generation with Langfuse."
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens; defaults to config.yaml.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit the number of dataset items evaluated.",
    )
    parser.add_argument(
        "--no-experiment",
        action="store_true",
        help="Run locally without creating a Langfuse experiment.",
    )

    args = parser.parse_args()

    if args.top_chapters < 1 or args.route_pool < 1:
        parser.error("--top-chapters and --route-pool must be positive.")

    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")

    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be positive.")

    return args


def main():
    args = parse_args()
    langfuse = get_client()
    qdrant_client = None

    try:
        if not langfuse.auth_check():
            raise RuntimeError("Langfuse authentication failed.")

        config = load_config()
        retrieval_config = config["retrieval"]

        # Use the same Qdrant path as evaluate_retrieval.py.
        persist_dir = config["index"]["persist_dir"] + "_qdrant"
        collection_name = config["index"]["collection_name"]

        mode = retrieval_config["mode"]
        top_k = retrieval_config["top_k"]
        candidate_k = retrieval_config["candidate_k"]

        max_new_tokens = (
            args.max_new_tokens
            if args.max_new_tokens is not None
            else config["generation"]["max_new_tokens"]
        )
        config["generation"]["max_new_tokens"] = max_new_tokens

        dataset = langfuse.get_dataset(args.dataset)
        items = list(dataset.items)

        if args.max_items is not None:
            items = items[:args.max_items]

        if not items:
            raise RuntimeError(
                f"Dataset '{args.dataset}' contains no items."
            )

        print(f"Dataset: {dataset.name}")
        print(f"Available items: {len(dataset.items)}")
        print(f"Items selected: {len(items)}")
        print(f"Retrieval mode: {mode}")
        print(
            f"Chapter routing: "
            f"{'enabled' if args.route else 'disabled'}"
        )
        print(f"Max new tokens: {max_new_tokens}")
        print(
            "Execution mode: "
            + (
                "LOCAL TEST (no Langfuse experiment)"
                if args.no_experiment
                else "LANGFUSE EXPERIMENT"
            )
        )

        # Load model once.
        model, tokenizer = load_llm(config)

        if args.route:
            index = load_index(
                persist_dir=persist_dir,
                collection_name=collection_name,
                embed_model=load_embedding_model(config["embedding"]),
            )
            qdrant_client = get_qdrant_client(persist_dir)
            retriever = None
        else:
            index = None
            retriever = load_retriever(
                persist_dir=persist_dir,
                collection_name=collection_name,
                embed_model=load_embedding_model(config["embedding"]),
                k=top_k,
                mode=mode,
                candidate_k=candidate_k,
            )

        def generation_task(*, item, **kwargs):
            question = item.input["question"]

            if args.route:
                routed = route_chapters(
                    index=index,
                    query=question,
                    top_n=args.top_chapters,
                    pool=args.route_pool,
                )
                chapters = [chapter for chapter, _ in routed]

                retriever_kwargs = {
                    "index": index,
                    "persist_dir": persist_dir,
                    "collection_name": collection_name,
                    "k": top_k,
                    "mode": mode,
                    "candidate_k": candidate_k,
                }
                if chapters:
                    retriever_kwargs["chapter"] = chapters

                question_retriever = build_retriever(**retriever_kwargs)
            else:
                question_retriever = retriever

            result = generate_answer(
                question=question,
                retriever=question_retriever,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )

            sources = result["sources"]
            retrieved_articles = sorted(
                {
                    str(source["article"])
                    for source in sources
                    if source.get("article") is not None
                }
            )

            return {
                "question": question,
                "answer": result["answer"],
                "generated_answer": result["answer"],
                "retrieved_articles": retrieved_articles,
                "sources": sources,
                "escaped": result.get("escaped"),
            }

        if args.no_experiment:
            hits = 0

            for number, item in enumerate(items, start=1):
                print(f"\n{'=' * 60}")
                print(f"Question {number}/{len(items)}")

                output = generation_task(item=item)
                expected_output = item.expected_output or {}
                evaluation = article_hit_evaluator(
                    output=output,
                    expected_output=expected_output,
                )

                hits += int(evaluation.value == 1.0)

                print("\nQuestion:")
                print(output["question"])
                print("\nGenerated answer:")
                print(output["generated_answer"])
                print("\nExpected articles:")
                print(expected_output.get("articles", []))
                print("\nRetrieved articles:")
                print(output["retrieved_articles"])
                print(f"\nArticle hit: {evaluation.value}")

                if output["escaped"]:
                    print("\nEscaped:", output["escaped"])

            total = len(items)
            print(f"\n{'=' * 60}")
            print("LOCAL TEST SUMMARY")
            print(f"Questions evaluated: {total}")
            print(f"Article hits: {hits}")
            print(f"Article misses: {total - hits}")
            print(f"Article hit rate: {hits / total:.3f}")
            print("No Langfuse experiment was created.")

        else:
            # Create a temporary Langfuse dataset containing only the
            # selected items when --max-items is used. This prevents
            # accidentally running the full dataset.
            experiment_dataset = dataset

            if args.max_items is not None:
                temporary_name = (
                    f"{args.dataset}_generation_subset_"
                    f"{args.max_items}"
                )
                experiment_dataset = langfuse.create_dataset(
                    name=temporary_name,
                    description=(
                        f"Temporary subset for generation evaluation "
                        f"from {args.dataset}; first {len(items)} items."
                    ),
                )

                for item in items:
                    experiment_dataset.create_item(
                        input=item.input,
                        expected_output=item.expected_output,
                        metadata=getattr(item, "metadata", None),
                    )

            result = experiment_dataset.run_experiment(
                name=(
                    f"Generation routed - top {args.top_chapters} chapters"
                    if args.route
                    else "Generation baseline"
                ),
                description=(
                    "Evaluate generated answers and expected article "
                    "retrieval. Chapter routing is "
                    f"{'enabled' if args.route else 'disabled'}. "
                    f"Source dataset: {args.dataset}. "
                    f"Items selected: {len(items)}. "
                    f"Max new tokens: {max_new_tokens}."
                ),
                task=generation_task,
                evaluators=[article_hit_evaluator, article_recall_evaluator],
            )

            print("\nEXPERIMENT RESULTS")
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