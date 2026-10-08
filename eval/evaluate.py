"""Recall@k of the retriever against eval/golden_set.json.

A question is a hit when at least one of the k retrieved chunks comes from
one of the pages listed for it. Questions with an empty "pages" list are
skipped (not labeled yet). Run from the repo root.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

K = 5  # same k that generation feeds to the model


def main():
    golden = json.loads((ROOT / "eval" / "golden_set.json").read_text())
    labeled = [g for g in golden if g["pages"]]
    print(f"{len(labeled)} labeled / {len(golden)} total questions")
    if not labeled:
        return

    # heavy imports and model loading only happen when there is something to measure
    from config import load_config
    from retrieval import load_retriever, retrieve

    config = load_config()
    retriever = load_retriever(persist_dir=config["index"]["persist_dir"], k=K)

    by_type = defaultdict(lambda: [0, 0])  # type -> [hits, total]
    misses = []
    for item in labeled:
        got = [r["page"] for r in retrieve(retriever, item["question"])]
        hit = bool({str(p) for p in item["pages"]} & set(got))
        by_type[item["type"]][1] += 1
        by_type[item["type"]][0] += hit
        if not hit:
            misses.append((item["question"], item["pages"], got))

    hits = sum(h for h, _ in by_type.values())
    total = sum(t for _, t in by_type.values())
    print(f"\nrecall@{K}: {hits}/{total} = {hits / total:.0%}")
    for t, (h, n) in sorted(by_type.items()):
        print(f"  {t}: {h}/{n} = {h / n:.0%}")

    for question, expected, got in misses:
        print(f"\nMISS: {question}\n  expected pages: {expected}\n  got pages: {got}")


if __name__ == "__main__":
    main()
