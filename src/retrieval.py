import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient, models
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_client(persist_dir: str) -> QdrantClient:
    """Return one shared client per directory (local mode allows only one)."""
    return QdrantClient(path=persist_dir)


def load_index(
    persist_dir: str,
    collection_name: str,
    model_name: str,
) -> VectorStoreIndex:
    """Open the existing Qdrant collection as a LlamaIndex index."""
    # Retrieval only: stop LlamaIndex from trying to load a default LLM.
    Settings.llm = MockLLM()

    embed_model = HuggingFaceEmbedding(model_name=model_name)

    vector_store = QdrantVectorStore(
        client=get_client(persist_dir),
        collection_name=collection_name,
    )
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )


# ---------------------------------------------------------------------------
# Stage 1: chapter routing
# ---------------------------------------------------------------------------

def route_chapters(
    index: VectorStoreIndex,
    query: str,
    top_n: int = 2,
    pool: int = 30,
) -> list[tuple[str, float]]:
    """
    Pick the chapters most relevant to the query.

    Runs an unfiltered dense search over the articles, then ranks each
    chapter by the score of its best article.

    Returns:
        A list of (chapter, score), best first, at most top_n items.
    """
    retriever = index.as_retriever(similarity_top_k=pool)

    best: dict[str, float] = {}
    for result in retriever.retrieve(query):
        chapter = result.node.metadata.get("chapter")
        if chapter is None:
            continue

        chapter = str(chapter)
        score = result.score or 0.0
        best[chapter] = max(best.get(chapter, float("-inf")), score)

    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Stage 2: retrieval inside the chosen chapters
# ---------------------------------------------------------------------------

def normalize_chapters(chapter: str | list[str] | None) -> list[str] | None:
    """Accept one chapter or a list; always return a list (or None)."""
    if chapter is None:
        return None
    if isinstance(chapter, str):
        return [chapter]
    return [str(c) for c in chapter] or None


def build_filters(chapters: list[str] | None) -> MetadataFilters | None:
    if not chapters:
        return None
    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="chapter",
                value=chapters,
                operator=FilterOperator.IN,
            )
        ]
    )


def load_nodes(
    client: QdrantClient,
    collection_name: str,
    chapters: list[str] | None,
) -> list[TextNode]:
    """Read the nodes (optionally only from some chapters) to build BM25."""
    scroll_filter = None
    if chapters:
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="chapter",
                    match=models.MatchAny(any=chapters),
                )
            ]
        )

    nodes: list[TextNode] = []
    skipped = 0
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            node_data = (point.payload or {}).get("_node_content")
            if not node_data:
                skipped += 1
                continue
            if isinstance(node_data, str):
                node_data = json.loads(node_data)
            nodes.append(TextNode.from_dict(node_data))

        if offset is None:
            break

    if skipped:
        print(f"Warning: {skipped} points without _node_content were skipped.")

    return nodes


def build_retriever(
    index: VectorStoreIndex,
    persist_dir: str,
    collection_name: str,
    k: int,
    mode: str,
    candidate_k: int,
    chapter: str | list[str] | None = None,
):
    """
    Build a retriever, optionally restricted to one or more chapters.

    Args:
        index: Index returned by load_index.
        persist_dir: Local Qdrant directory (to reuse the shared client).
        collection_name: Qdrant collection name.
        k: Number of final results.
        mode: "dense" or "hybrid".
        candidate_k: Candidate pool size for hybrid retrieval.
        chapter: A chapter, a list of chapters, or None for no filter.
    """
    if mode not in ("dense", "hybrid"):
        raise ValueError("mode must be either 'dense' or 'hybrid'.")

    chapters = normalize_chapters(chapter)
    filters = build_filters(chapters)

    if mode == "dense":
        return index.as_retriever(similarity_top_k=k, filters=filters)

    nodes = load_nodes(get_client(persist_dir), collection_name, chapters)
    if not nodes:
        raise ValueError(f"No text nodes found (chapters={chapters!r}).")

    pool = max(candidate_k, k)

    vector_retriever = index.as_retriever(
        similarity_top_k=pool,
        filters=filters,
    )
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=min(pool, len(nodes)),
    )

    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )


def retrieve(query: str, retriever) -> list[dict]:
    """Retrieve relevant chunks and return their text and metadata."""
    output = []
    for result in retriever.retrieve(query):
        node = result.node
        metadata = node.metadata or {}
        output.append(
            {
                "text": node.get_content(),
                "article": metadata.get("article"),
                "chapter": metadata.get("chapter"),
                "chapter_title": metadata.get("chapter_title"),
                "score": result.score,
            }
        )
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run retrieval against the Qdrant index."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the prohibited AI practices?",
        help="Search query.",
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid"],
        default=None,
        help="Retrieval mode. Defaults to the config setting.",
    )
    parser.add_argument(
        "--chapter",
        nargs="+",
        default=None,
        help="Manual chapter filter, e.g. --chapter II III.",
    )
    parser.add_argument(
        "--route",
        action="store_true",
        help="Pick chapters automatically before searching (stage 1).",
    )
    parser.add_argument(
        "--top-chapters",
        type=int,
        default=2,
        help="How many chapters to keep when using --route.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full text instead of the first 500 characters.",
    )
    args = parser.parse_args()

    if args.route and args.chapter:
        parser.error("Use either --route or --chapter, not both.")

    config = load_config()

    mode = args.mode or config["retrieval"]["mode"]
    top_k = (
        args.top_k if args.top_k is not None else config["retrieval"]["top_k"]
    )
    persist_dir = config["index"]["persist_dir"] + "_qdrant"
    collection_name = config["index"]["collection_name"]

    try:
        index = load_index(
            persist_dir=persist_dir,
            collection_name=collection_name,
            model_name=config["embedding"]["model_name"],
        )

        chapters = args.chapter
        routed = None
        if args.route:
            routed = route_chapters(
                index, args.query, top_n=args.top_chapters
            )
            chapters = [c for c, _ in routed]

        retriever = build_retriever(
            index=index,
            persist_dir=persist_dir,
            collection_name=collection_name,
            k=top_k,
            mode=mode,
            candidate_k=config["retrieval"]["candidate_k"],
            chapter=chapters,
        )
        results = retrieve(args.query, retriever)
    finally:
        get_client(persist_dir).close()

    print(f"\nQuery: {args.query}")
    print(f"Mode: {mode}")
    if routed is not None:
        print("Routed chapters: " + ", ".join(f"{c} ({s:.3f})" for c, s in routed))
    else:
        print(f"Chapter filter: {chapters or 'None'}")
    print(f"Results: {len(results)}")

    for i, result in enumerate(results, start=1):
        text = result["text"] if args.full else result["text"][:500]
        print(f"\n--- Result {i} ---")
        print(f"Chapter: {result['chapter']}")
        print(f"Article: {result['article']}")
        print(f"Chapter title: {result['chapter_title']}")
        print(f"Score: {result['score']}")
        print(text)


if __name__ == "__main__":
    main()