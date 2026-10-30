
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
        help="Maximum number of dataset items to evaluate.",
    )

    args = parser.parse_args()

    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")

    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be positive.")

    return args


def main():
    args = parse_args()

    config_path = ROOT_DIR / "configs" / "config.yaml"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.max_new_tokens is not None:
        config["generation"]["max_new_tokens"] = args.max_new_tokens

    # Load the existing Qdrant collection. Do not rebuild the index.
    retriever = load_retriever(
        persist_dir=config["index"]["persist_dir"],
        collection_name=config["index"]["collection_name"],
        model_name=config["embedding"]["model_name"],
        k=config["retrieval"]["top_k"],
        mode=config["retrieval"]["mode"],
        candidate_k=config["retrieval"]["candidate_k"],
    )

    model, tokenizer = load_llm(config)
    langfuse = get_client()

    try:
        if not langfuse.auth_check():
            raise RuntimeError("Langfuse authentication failed.")

        dataset = langfuse.get_dataset(args.dataset)

        print(f"Dataset: {dataset.name}")
        print(f"Available items: {len(dataset.items)}")
        print(
            "Max new tokens: "
            f"{config['generation']['max_new_tokens']}"
        )

        def generation_task(*, item, **kwargs):
            question = item.input["question"]

            result = generate_answer(
                question=question,
                retriever=retriever,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config["generation"]["max_new_tokens"],
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
                    f"Expected: {sorted(expected)}; "
                    f"Retrieved: {sorted(retrieved)}"
                ),
            )

        # Important: run_experiment executes the dataset items.
        # This implementation uses max_items only if the Langfuse SDK
        # supports passing a subset of dataset items to run_experiment.
        # Otherwise, it runs the entire dataset.
        if args.max_items is not None:
            print(
                f"Requested item limit: {args.max_items}. "
                "The dataset may still run in full depending on "
                "your Langfuse SDK's run_experiment API."
            )

        experiment_kwargs = {
            "name": "Generation evaluation",
            "description": (
                "Evaluate RAG generation and expected article retrieval. "
                f"Dataset: {args.dataset}."
            ),
            "task": generation_task,
            "evaluators": [article_hit_evaluator],
        }

        result = dataset.run_experiment(**experiment_kwargs)

        print(result.format())

        experiment_url = getattr(result, "experiment_url", None)
        if experiment_url:
            print("Experiment:", experiment_url)

    finally:
        langfuse.flush()


if __name__ == "__main__":
    main()
