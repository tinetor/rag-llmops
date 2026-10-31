"""Shared evaluators for retrieval and generation experiments."""

from langfuse import Evaluation


def article_hit_evaluator(*, output, expected_output, **kwargs):
    """Return 1.0 if at least one expected article was retrieved."""
    expected = set(
        map(str, (expected_output or {}).get("articles", []))
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

def article_recall_evaluator(*, output, expected_output, **kwargs):
    """Return the fraction of expected articles that were retrieved."""
    expected = set(
        map(str, (expected_output or {}).get("articles", []))
    )
    retrieved = set(output["retrieved_articles"])
    found = expected & retrieved
    missing = expected - retrieved

    recall = len(found) / len(expected)     
    if not expected:
        raise ValueError(
            "Item has no expected articles; check the dataset."
        )

    return Evaluation(
        name="article_recall",
        value=recall,
        comment=(
            f"Found {len(found)}/{len(expected)}; "
            f"Missing: {sorted(missing)}; "
            f"Retrieved: {sorted(retrieved)}"
        ),
    )