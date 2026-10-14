"""Evaluate article retrieval, save experiment results, and compare with baseline."""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

GOLDEN_PATH = ROOT / "eval" / "golden_set.json"
RESULTS_PATH = ROOT / "eval" / "results.jsonl"
BASELINE_PATH = ROOT / "eval" / "baseline.json"

ARTICLE_PATTERN = re.compile(
    r"\b(?:Art(?:icle)?\.?)\s*(\d+)\b",
    re.IGNORECASE,
)


def expected_articles(item):
    """Extract article numbers from the hint field."""
    return set(ARTICLE_PATTERN.findall(item.get("hint", "")))


def retrieved_articles(results):
    """Collect article IDs from retrieved chunk metadata."""
    return {
        str(article)
        for result in results
        if (article := result.get("article")) is not None
    }


def main():
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    labeled = [item for item in golden if expected_articles(item)]

    print(f"Questions in golden set: {len(golden)}")
    print(f"Questions labeled by article: {len(labeled)}")

    if not labeled:
        print("No questions with article labels to evaluate.")
        return

    from config import load_config
    from retrieval import load_retriever, retrieve

    config = load_config()
    retrieval_config = config["retrieval"]
    k = retrieval_config["top_k"]

    retriever = load_retriever(
        persist_dir=config["index"]["persist_dir"],
        model_name=config["embedding"]["model_name"],
        k=k,
        mode=retrieval_config["mode"],
        candidate_k=retrieval_config["candidate_k"],
    )

    stats = defaultdict(lambda: [0, 0])
    question_results = []
    misses = []
    unmeasurable = []

    for item in labeled:
        expected = expected_articles(item)
        results = retrieve(retriever, item["question"])
        got = retrieved_articles(results)
        question_type = item.get("type", "unknown")

        # No article metadata counts as a retrieval failure.
        hit = bool(expected & got)

        if not got:
            unmeasurable.append(item["question"])

        stats[question_type][0] += int(hit)
        stats[question_type][1] += 1

        question_results.append({
            "question": item["question"],
            "type": question_type,
            "expected_articles": sorted(expected),
            "retrieved_articles": sorted(got),
            "hit": hit,
            "measurable": bool(got),
        })

        # Save the retrieved chunk contents only for failed questions.
        if not hit:
            miss = {
                "question": item["question"],
                "expected": sorted(expected),
                "retrieved": sorted(got),
                "has_article_metadata": bool(got),
                "retrieved_chunks": [
                    {
                        "article": result.get("article"),
                        "page": result.get("page"),
                        "score": result.get("score"),
                        "text": result.get("text"),
                    }
                    for result in results
                ],
            }
            misses.append(miss)

            print(f"\nARTICLE MISS: {miss['question']}")
            print(f"  Expected articles: {miss['expected']}")
            print(f"  Retrieved articles: {miss['retrieved']}")

            if not miss["has_article_metadata"]:
                print("  Reason: no retrieved article metadata")

            for i, chunk in enumerate(miss["retrieved_chunks"], start=1):
                print(
                    f"\n  Chunk {i} | Article: {chunk['article']} "
                    f"| Page: {chunk['page']} | Score: {chunk['score']}"
                )
                print(f"  {chunk['text']}")

    hits = sum(h for h, _ in stats.values())
    total = sum(n for _, n in stats.values())
    recall = hits / total if total else None

    by_type = {
        name: {
            "hits": h,
            "total": n,
            "recall": h / n if n else None,
        }
        for name, (h, n) in sorted(stats.items())
    }

    print(f"\nRECALL BY ARTICLE (k={k})")
    if recall is not None:
        print(f"Recall@{k}: {hits}/{total} = {recall:.1%}")
    else:
        print("Not measurable: no labeled questions.")

    for name, metric in by_type.items():
        print(
            f"  {name}: {metric['hits']}/{metric['total']} "
            f"= {metric['recall']:.1%}"
        )

    print(
        f"\nQuestions without retrieved article metadata: "
        f"{len(unmeasurable)}"
    )

    question_ids = sorted(item["question"] for item in labeled)

    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "chunking": config.get("chunking", {}),
            "embedding": config.get("embedding", {}),
            "retrieval": retrieval_config,
        },
        "evaluation": {
            "metric": "recall_by_article",
            "k": k,
            "golden_set_questions": len(golden),
            "labeled_questions": len(labeled),
            "evaluated_questions": total,
            "unmeasurable_questions": len(unmeasurable),
            "hits": hits,
            "recall": recall,
            "by_type": by_type,
        },
        "question_ids": question_ids,
        "unmeasurable_questions": unmeasurable,
        "misses": misses,
        "question_results": question_results,
    }

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not BASELINE_PATH.exists():
        baseline = {
            **run,
            "baseline_note": (
                "Created using the current evaluation rules. "
                "Questions without article metadata count as misses."
            ),
        }
        BASELINE_PATH.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        comparison = {
            "status": "BASELINE_CREATED",
            "message": "This run is now the baseline.",
        }
    else:
        baseline = json.loads(
            BASELINE_PATH.read_text(encoding="utf-8")
        )
        base_eval = baseline["evaluation"]

        same_questions = question_ids == baseline.get("question_ids")
        same_k = k == base_eval.get("k")
        compatible_rules = (
            "measurable" in baseline.get("question_results", [{}])[0]
            and all(
                result.get("hit") is not None
                for result in baseline.get("question_results", [])
            )
        )

        if not same_questions or not same_k or not compatible_rules:
            reasons = []

            if not same_questions:
                reasons.append("question sets differ")
            if not same_k:
                reasons.append("top_k differs")
            if not compatible_rules:
                reasons.append(
                    "baseline uses different evaluation rules; recreate baseline"
                )

            comparison = {
                "status": "INCONCLUSIVE",
                "reasons": reasons,
                "baseline_recall": base_eval.get("recall"),
                "current_recall": recall,
            }
        else:
            delta = recall - base_eval["recall"]
            comparison = {
                "status": (
                    "BETTER" if delta > 0
                    else "WORSE" if delta < 0
                    else "UNCHANGED"
                ),
                "baseline_recall": base_eval["recall"],
                "current_recall": recall,
                "delta_percentage_points": round(delta * 100, 2),
            }

    run["comparison"] = comparison

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(run, ensure_ascii=False) + "\n")

    chunking = run["config"]["chunking"]
    embedding = run["config"]["embedding"]

    print("\n" + "=" * 52)
    print("EXPERIMENT SUMMARY")
    print("=" * 52)
    print(f"Strategy:       {chunking.get('strategy', 'N/A')}")
    print(f"Chunk size:     {chunking.get('max_chars', 'N/A')}")
    print(f"Overlap:        {chunking.get('overlap_ratio', 'N/A')}")
    print(f"Embedding:      {embedding.get('model_name', 'N/A')}")
    print(f"Retrieval:      {retrieval_config.get('mode', 'N/A')}")
    print(f"Top k:          {k}")
    print(f"Candidate k:    {retrieval_config.get('candidate_k', 'N/A')}")
    print(f"Evaluated:      {total}/{len(labeled)}")

    if recall is not None:
        print(f"Recall@{k}:       {hits}/{total} = {recall:.1%}")
    else:
        print(f"Recall@{k}:       N/A")

    print(f"Unmeasurable:   {len(unmeasurable)}")
    print(f"Failed queries: {len(misses)}")
    print(f"Result:         {comparison['status']}")

    if comparison.get("baseline_recall") is not None:
        print(f"Baseline:       {comparison['baseline_recall']:.1%}")

    if "delta_percentage_points" in comparison:
        print(
            "Difference:     "
            f"{comparison['delta_percentage_points']:+.2f} percentage points"
        )

    if comparison.get("reasons"):
        print("Reason:         " + "; ".join(comparison["reasons"]))

    print(f"Results saved:  {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Baseline file:  {BASELINE_PATH.relative_to(ROOT)}")
    print("=" * 52)


if __name__ == "__main__":
    main()