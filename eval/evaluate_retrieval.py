"""Evaluate article retrieval using a Langfuse dataset experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from langfuse import Evaluation, get_client
from config import load_config
from retrieval import load_retriever, retrieve

DATASET_NAME = "Golden_Dataset"


def main():
    langfuse = get_client()

    if not langfuse.auth_check():
        raise RuntimeError("Langfuse authentication failed.")

    config = load_config()
    retrieval_config = config["retrieval"]

    retriever = load_retriever(
        persist_dir=config["index"]["persist_dir"],
        model_name=config["embedding"]["model_name"],
        k=retrieval_config["top_k"],
        mode=retrieval_config["mode"],
        candidate_k=retrieval_config["candidate_k"],
    )

    dataset = langfuse.get_dataset(DATASET_NAME)
    print(f"Dataset: {dataset.name} | Items: {len(dataset.items)}")

    def retrieval_task(*, item, **kwargs):
        question = item.input["question"]
        chunks = retrieve(retriever, question)

        articles = sorted({
            str(chunk["article"])
            for chunk in chunks
            if chunk.get("article") is not None
        })

        return {
            "question": question,
            "retrieved_articles": articles,
            "chunks": chunks,
        }

    def article_hit_evaluator(*, output, expected_output, **kwargs):
        expected = set(map(str, expected_output["articles"]))
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

    try:
        result = dataset.run_experiment(
            name="Retrieval baseline",
            description=(
                "Evaluate whether expected articles appear "
                "in the retrieved chunks."
            ),
            task=retrieval_task,
            evaluators=[article_hit_evaluator],
        )

        print(result.format())

        if getattr(result, "experiment_url", None):
            print("Experiment:", result.experiment_url)

    finally:
        langfuse.flush()


if __name__ == "__main__":
    main()
