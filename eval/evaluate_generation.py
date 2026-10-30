
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
        help=(
            "Maximum number of tokens generated per answer. "
            "If omitted, uses the value in config.yaml."
        ),
    )

    args = parser.parse_args()

    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be a positive integer.")

    return args


def main():
    args = parse_args()

    config_path = ROOT_DIR / "configs" / "config.yaml"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Override max_new_tokens only when specified through Bash.
    if args.max_new_tokens is not None:
        config["generation"]["max_new_tokens"] = args.max_new_tokens

    persist_dir = config["index"]["persist_dir"]
    collection_name = config["index"].get("collection_name")

    retriever_kwargs = {
        "persist_dir": persist_dir,
        "model_name": config["embedding"]["model_name"],
        "k": config["retrieval"]["top_k"],
        "mode": config["retrieval"]["mode"],
        "candidate_k": config["retrieval"]["candidate_k"],
    }

    # Pass collection_name only if configured and supported by your
    # retrieval.load_retriever implementation.
    if collection_name is not None:
        retriever_kwargs["collection_name"] = collection_name

    retriever = load_retriever(**retriever_kwargs)

    model, tokenizer = load_llm(config)

    langfuse = get_client()

    try:
        if not langfuse.auth_check():
            raise RuntimeError("Langfuse authentication failed.")

        dataset = langfuse.get_dataset(args.dataset)

        print(f"Dataset: {dataset.name}")
        print(f"Items: {len(dataset.items)}")
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

        result = dataset.run_experiment(
            name="Generation evaluation",
            description=(
                "Evaluate generated answers and whether the expected "
                "articles appear among the retrieved sources. "
                f"Dataset: {args.dataset}."
            ),
            task=generation_task,
            evaluators=[article_hit_evaluator],
        )

        print(result.format())

        if getattr(result, "experiment_url", None):
            print("Experiment:", result.experiment_url)

    finally:
        langfuse.flush()


if __name__ == "__main__":
    main()
