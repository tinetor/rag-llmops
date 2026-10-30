
"""Evaluate RAG generation using a Langfuse dataset experiment."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from langfuse import Evaluation, get_client
from config import load_config
from generation import generate_answer
from llm import load_llm
from retrieval import (
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
        help="Run a local test on the first N dataset items without creating a Langfuse experiment.",
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

        # Match the Qdrant path used by evaluate.retrieval.py.
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

        # Preserve the configured generation parameters, with an optional CLI override.
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
        print(f"Chapter routing: {'enabled' if args.route else 'disabled'}")
        print(f"Max new tokens: {max_new_tokens}")
        print(
            "Execution mode: "
            + (
                "LOCAL TEST (no Langfuse experiment)"
                if args.max_items is not None
                else "FULL LANGFUSE EXPERIMENT"
            )
        )

        # Load the model once for the entire evaluation.
        model, tokenizer = load_llm(config)

        if args.route:
            index = load_index(
                persist_dir=persist_dir,
                collection_name=collection_name,
                model_name=config["embedding"]["model_name"],
            )
            qdrant_client = get_qdrant_client(persist_dir)
            retriever = None
        else:
            index = None
            retriever = load_retriever(
                persist_dir=persist_dir,
                collection_name=collection_name,
                model_name=config["embedding"]["model_name"],
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

                if chapters:
                    question_retriever = build_retriever(
                        index=index,
                        persist_dir=persist_dir,
                        collection_name=collection_name,
                        k=top_k,
                        mode=mode,
                        candidate_k=candidate_k,
                        chapter=chapters,
                    )
                else:
                    question_retriever = build_retriever(
                        index=index,
                        persist_dir=persist_dir,
                        collection_name=collection_name,
                        k=top_k,
                        mode=mode,
                        candidate_k=candidate_k,
                    )

                # Use the same retrieval helper as the retrieval evaluation.
                chunks = retrieve(question_retriever, question)

                # Generate using the same retrieved chunks so that the
                # routed retrieval result is also the generation context.
                result = generate_answer(
                    question=question,
                    retriever=question_retriever,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                )
            else:
                result = generate_answer(
                    question=question,
                    retriever=retriever,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                )
                chunks = result["sources"]

            retrieved_articles = sorted(
                {
                    str(source["article"])
                    for source in chunks
                    if source.get("article") is not None
                }
            )

            return {
                "question": question,
                "answer": result["answer"],
                "retrieved_articles": retrieved_articles,
                "sources": chunks,
                "escaped": result.get("escaped"),
            }

        def article_hit_evaluator(*, output, expected_output, **kwargs):
            expected = set(
                map(str, expected_output.get("articles", []))
            )
            retrieved = set(output["retrieved_articles"])
            hit = bool(expected & retrieved)

            return Evaluation(
                name="article_hit",
                value=1.0 if hit else 0.0,
                comment=(
                    f"Expected: {sorted(expected)}; "
                    f"Retrieved: {sorted(retrieved)}"
                ),
            )

        if args.max_items is not None:
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
                print(output["answer"])
                print("\nExpected articles:")
                print(expected_output.get("articles", []))
                print("\nRetrieved articles:")
                print(output["retrieved_articles"])
                print(f"\nArticle hit: {evaluation.value}")

                if output["escaped"]:
                    print("\nEscaped:", output["escaped"])

            print(f"\n{'=' * 60}")
            print("LOCAL TEST SUMMARY")
            print(f"Questions evaluated: {len(items)}")
            print(f"Article hits: {hits}")
            print(f"Article misses: {len(items) - hits}")
            print(f"Article hit rate: {hits / len(items):.3f}")
            print("No Langfuse experiment was created.")

        else:
            result = dataset.run_experiment(
                name=(
                    f"Generation routed - top {args.top_chapters} chapters"
                    if args.route
                    else "Generation baseline"
                ),
                description=(
                    "Evaluate generated answers and expected article "
                    "retrieval. Chapter routing is "
                    f"{'enabled' if args.route else 'disabled'}. "
                    f"Dataset: {args.dataset}. "
                    f"Max new tokens: {max_new_tokens}."
                ),
                task=generation_task,
                evaluators=[article_hit_evaluator],
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
