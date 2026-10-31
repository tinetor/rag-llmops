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