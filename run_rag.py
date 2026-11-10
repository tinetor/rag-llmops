"""Command-line entry point for the RAG application."""

import argparse


from rag_llmops.rag_pipeline import run_rag


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question to the RAG application."
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask the RAG application.",
    )
    args = parser.parse_args()

    result = run_rag(args.question)

    print(f"Q: {args.question}\n")
    print(f"A: {result['answer']}\n")
    print(f"Escaped: {result['escaped']}")
    print("Sources:")

    for source in result["sources"]:
        score = (
            f"{source['score']:.3f}"
            if source["score"] is not None
            else "N/A"
        )
        article = source["article"]
        article_label = (
            f"Article {article}"
            if article is not None
            else "Article unknown"
        )

        print(f"  [{score}] {article_label}")
        print(f"    {source['text'][:120]!r}")


if __name__ == "__main__":
    main()
