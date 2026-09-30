"""Evaluates retrieval quality (recall@k) against the golden set.
This is the script that later becomes the CI regression gate."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chunking import load_config
from retrieval import load_retriever, retrieve

RECALL_THRESHOLD = 0.8  # minimum acceptable recall@k before the pipeline is considered broken


def recall_at_k(golden_set, model, index, chunks, k):
    hits = 0
    misses = []
    for item in golden_set:
        results = retrieve(item["question"], model, index, chunks, k=k)
        retrieved_headings = [r["heading"] for r in results]
        if any(item["relevant_heading"] in h for h in retrieved_headings):
            hits += 1
        else:
            misses.append((item["question"], item["relevant_heading"], retrieved_headings))
    return hits / len(golden_set), misses


if __name__ == "__main__":
    config = load_config()
    golden_set = json.loads(Path("eval/golden_set.json").read_text())
    model, index, chunks = load_retriever(config)

    score, misses = recall_at_k(golden_set, model, index, chunks, k=config["retrieval"]["k"])

    print(f"Recall@{config['retrieval']['k']}: {score:.2%} ({len(golden_set)} questions)\n")
    for question, expected, got in misses:
        print(f"MISS: '{question}' -> expected '{expected}', got {got}")

    if score < RECALL_THRESHOLD:
        print(f"\nFAIL: recall {score:.2%} is below threshold {RECALL_THRESHOLD:.0%}")
        sys.exit(1)
    else:
        print(f"\nPASS: recall {score:.2%} meets threshold {RECALL_THRESHOLD:.0%}")
        sys.exit(0)
