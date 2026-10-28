import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from retrieval import load_retriever, retrieve

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.retrievers.bm25 import BM25Retriever


def parse_args(config):
    parser = argparse.ArgumentParser(
        description="Compare dense, BM25, and hybrid retrieval."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question to test.",
    )
    parser.add_argument(
        "--expected-article",
        required=True,
        help="Expected article number.",
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid", "all"],
        default="all",
        help="Retrieval method to run (default: all).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config["retrieval"]["top_k"],
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=config["retrieval"]["candidate_k"],
    )
    parser.add_argument(
        "--embedding",
        default=config["embedding"]["model_name"],
    )

    return parser.parse_args()


def print_results(name, results, expected_article):
    print(f"\n{'=' * 15} {name.upper()} {'=' * 15}")

    found = False

    for rank, result in enumerate(results, start=1):
        article = result["article"]
        article = str(article) if article is not None else "unknown"

        if article == str(expected_article):
            found = True

        score = result["score"]
        score = f"{score:.4f}" if score is not None else "N/A"

        print(
            f"{rank}. Article {article} | score={score} "
           
        )
        print(f"   {result['text'][:180]!r}")

    print(
        f"\nExpected article {expected_article} in top results: "
        f"{'YES' if found else 'NO'}"
    )


def load_index(persist_dir, model_name):
    embed_model = HuggingFaceEmbedding(model_name=model_name)

    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=persist_dir,
    )

    index = load_index_from_storage(
        storage_context,
        embed_model=embed_model,
    )

    return index


def load_bm25_retriever(index, top_k):
    nodes = [
        node
        for node in index.docstore.docs.values()
        if isinstance(node, TextNode)
    ]

    if not nodes:
        raise ValueError(
            "No TextNodes found in the persisted document store."
        )

    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
    )


def main():
    config = load_config()
    args = parse_args(config)

    persist_dir = config["index"]["persist_dir"]

    print(f"Question: {args.query}")
    print(f"Expected article: {args.expected_article}")
    print(f"Embedding: {args.embedding}")
    print(f"Top-k: {args.top_k}")
    print(f"Candidate-k: {args.candidate_k}")

    modes = (
        ["dense", "bm25", "hybrid"]
        if args.mode == "all"
        else [args.mode]
    )

    # Load the index once when BM25 is needed.
    index = None
    if "bm25" in modes:
        index = load_index(
            persist_dir=persist_dir,
            model_name=args.embedding,
        )

    for mode in modes:
        if mode == "bm25":
            retriever = load_bm25_retriever(
                index=index,
                top_k=args.top_k,
            )
        else:
            retriever = load_retriever(
                persist_dir=persist_dir,
                model_name=args.embedding,
                k=args.top_k,
                mode=mode,
                candidate_k=args.candidate_k,
            )

        results = retrieve(retriever, args.query)
        print_results(mode, results, args.expected_article)


if __name__ == "__main__":
    main()
