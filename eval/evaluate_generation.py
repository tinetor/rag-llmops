
"""Evaluate RAG generation using a Langfuse dataset experiment."""

import argparse
import sys
from pathlib import Path

import yaml
from langfuse import Evaluation, get_client

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generation import generate_answer
from llm import load_llm
from retrieval import load_retriever


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
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum tokens generated per answer. Defaults to config.yaml.",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help=(
            "Limit the number of questions. "
            "When specified, runs a local test without creating "
            "a Langfuse experiment."
        ),
    )

    args = parser.parse_args()

    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be a positive integer.")

    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be a positive integer.")

    return args


def main():
    args = parse_args()

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------
    config_path = ROOT_DIR / "configs" / "config.yaml"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.max_new_tokens is not None:
        config["generation"]["max_new_tokens"] = args.max_new_tokens

    max_new_tokens = config["generation"]["max_new_tokens"]

    # ---------------------------------------------------------
    # 2. Load the existing retriever and model
    # ---------------------------------------------------------
    # Uses the existing Qdrant collection; does not rebuild the index.
    retriever = load_retriever(
        persist_dir=config["index"]["persist_dir"],
        collection_name=config["index"]["collection_name"],
        model_name=config["embedding"]["model_name"],
        k=config["retrieval"]["top_k"],
        mode=config["retrieval"]["mode"],
        candidate_k=config["retrieval"]["candidate_k"],
    )

    model, tokenizer = load_llm(config)

    # ---------------------------------------------------------
    # 3. Connect to Langfuse and select dataset
    # ---------------------------------------------------------
    langfuse = get_client()

    try:
        if not langfuse.auth_check():
            raise RuntimeError("Langfuse authentication failed.")

        dataset = langfuse.get_dataset(args.dataset)

        items = list(dataset.items)

        if args.max_items is not None:
            items = items[:args.max_items]

        if not items:
            raise RuntimeError(
                f"Dataset '{args.dataset}' contains no items."
            )

        print("=" * 60)
        print("RAG GENERATION EVALUATION")
        print("=" * 60)
        print(f"Dataset: {dataset.name}")
        print(f"Available items: {len(dataset.items)}")
        print(f"Items selected: {len(items)}")
        print(f"Max new tokens: {max_new_tokens}")
        print(
            "Execution mode: "
            + (
                "LOCAL TEST (no Langfuse experiment)"
                if args.max_items is not None
                else "FULL LANGFUSE EXPERIMENT"
            )
        )
        print("=" * 60)

        # -----------------------------------------------------
        # 4. Generation task
        # -----------------------------------------------------
        def generation_task(*, item, **kwargs):
            question = item.input["question"]

            result = generate_answer(
                question=question,
                retriever=retriever,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )

            retrieved_articles = sorted(
                {
                    str(source["article"])
                    for source in result["sources"]
                    if source.get("article") is not None
                }
            )

            return {
                "question": question,
                "answer": result["answer"],
                "retrieved_articles": retrieved_articles,
                "sources": result["sources"],
                "escaped": result["escaped"],
            }

        # -----------------------------------------------------
        # 5. Retrieval article-hit evaluator
        # -----------------------------------------------------
        def article_hit_evaluator(
            *, output, expected_output, **kwargs
        ):
            expected = set(
                map(str, expected_output["articles"])
            )
            retrieved = set(output["retrieved_articles"])
            hit = bool(expected & retrieved)

            return Evaluation(
                name="article_hit",
                value=1.0 if hit else 0.0,
                comment=(
                    f"Expected articles: {sorted(expected)}; "
                    f"Retrieved articles: {sorted(retrieved)}"
                ),
            )

        # -----------------------------------------------------
        # 6. Local test mode: execute only selected items
        # -----------------------------------------------------
        if args.max_items is not None:
            hits = 0
            misses = 0

            for index, item in enumerate(items, start=1):
                print(f"\nQuestion {index}/{len(items)}")

                output = generation_task(item=item)
                expected_output = item.expected_output

                evaluation = article_hit_evaluator(
                    output=output,
                    expected_output=expected_output,
                )

                if evaluation.value == 1.0:
                    hits += 1
                else:
                    misses += 1

                print("\nQuestion:")
                print(output["question"])

                print("\nGenerated answer:")
                print(output["answer"])

                print("\nExpected articles:")
                print(expected_output.get("articles", []))

                print("\nRetrieved articles:")
                print(output["retrieved_articles"])

                print("\nArticle hit:")
                print(evaluation.value)

                print("\nSources:")
                for source in output["sources"]:
                    print(source)

                if output["escaped"]:
                    print("\nEscaped:", output["escaped"])

                print("-" * 60)

            total = hits + misses
            score = hits / total if total else 0.0

            print("\nLOCAL TEST SUMMARY")
            print(f"Questions evaluated: {total}")
            print(f"Article hits: {hits}")
            print(f"Article misses: {misses}")
            print(f"Article hit rate: {score:.3f}")
            print(
                "\nNo Langfuse experiment was created. "
                "This was a local test only."
            )

            return

        # -----------------------------------------------------
        # 7. Full dataset experiment in Langfuse
        # -----------------------------------------------------
        result = dataset.run_experiment(
            name="Generation evaluation",
            description=(
                "Evaluate generated answers and whether expected "
                "articles appear among retrieved sources. "
                f"Dataset: {args.dataset}. "
                f"Max new tokens: {max_new_tokens}."
            ),
            task=generation_task,
            evaluators=[article_hit_evaluator],
        )

        print("\nEXPERIMENT RESULTS")
        print(result.format())

        experiment_url = getattr(result, "experiment_url", None)

        if experiment_url:
            print("Experiment:", experiment_url)

    finally:
        langfuse.flush()


if __name__ == "__main__":
    main()
