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
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402


@lru_cache(maxsize=None)
def get_client(persist_dir: str) -> QdrantClient:
    """Return one shared client per local Qdrant directory."""
    return QdrantClient(path=persist_dir)


def load_index(
    persist_dir: str,
    collection_name: str,
    model_name: str,
) -> VectorStoreIndex:
    """Open the existing Qdrant collection as a LlamaIndex index."""
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


def route_chapters(
    index: VectorStoreIndex,
    query: str,
    top_n: int = 2,
    pool: int = 30,
) -> list[tuple[str, float]]:
    """Rank chapters by the best dense-retrieval score among their chunks."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if pool < 1:
        raise ValueError("pool must be at least 1.")

    retriever = index.as_retriever(similarity_top_k=pool)
    best_scores: dict[str, float] = {}

    for result in retriever.retrieve(query):
        chapter = (result.node.metadata or {}).get("chapter")
        if chapter is None:
            continue
        chapter = str(chapter)
        score = float(result.score or 0.0)
        best_scores[chapter] = max(best_scores.get(chapter, float("-inf")), score)

    ranked = sorted(best_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_n]


def normalize_chapters(chapter: str | list[str] | None) -> list[str] | None:
    """Normalize one chapter or a list of chapters."""
    if chapter is None:
        return None
    if isinstance(chapter, str):
        return [chapter]
    chapters = [str(value) for value in chapter]
    return chapters or None


def build_filters(chapters: list[str] | None) -> MetadataFilters | None:
    """Build a LlamaIndex metadata filter for the selected chapters."""
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
    """Load stored nodes for BM25, optionally restricted to selected chapters."""
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
        print(f"Warning: skipped {skipped} Qdrant points without _node_content.")
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
    """Build a dense or hybrid retriever, optionally filtered by chapter."""
    if mode not in ("dense", "hybrid"):
        raise ValueError("mode must be either 'dense' or 'hybrid'.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    chapters = normalize_chapters(chapter)
    filters = build_filters(chapters)

    if mode == "dense":
        return index.as_retriever(similarity_top_k=k, filters=filters)

    pool = max(candidate_k, k)
    nodes = load_nodes(get_client(persist_dir), collection_name, chapters)
    if not nodes:
        raise ValueError(f"No text nodes found for chapters={chapters!r}.")

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
    """Return retrieved chunks with relevant metadata."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve chunks from the existing Qdrant index."
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
        help="Manually filter chapters, e.g. --chapter II III.",
    )
    parser.add_argument(
        "--route",
        action="store_true",
        help="Automatically select chapters before retrieving chunks.",
    )
    parser.add_argument(
        "--top-chapters",
        type=int,
        default=2,
        help="Number of chapters selected by routing (default: 2).",
    )
    parser.add_argument(
        "--route-pool",
        type=int,
        default=30,
        help="Number of global dense results used for chapter routing (default: 30).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of final chunks to return.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full chunk text instead of the first 500 characters.",
    )
    args = parser.parse_args()

    if args.route and args.chapter:
        parser.error("Use either --route or --chapter, not both.")
    if args.top_chapters < 1:
        parser.error("--top-chapters must be at least 1.")
    if args.route_pool < 1:
        parser.error("--route-pool must be at least 1.")

    config = load_config()
    mode = args.mode or config["retrieval"]["mode"]
    top_k = args.top_k if args.top_k is not None else config["retrieval"]["top_k"]
    persist_dir = config["index"]["persist_dir"] + "_qdrant"
    collection_name = config["index"]["collection_name"]

    routed: list[tuple[str, float]] | None = None
    chapters = args.chapter
    results: list[dict] = []

    try:
        index = load_index(
            persist_dir=persist_dir,
            collection_name=collection_name,
            model_name=config["embedding"]["model_name"],
        )

        if args.route:
            routed = route_chapters(
                index=index,
                query=args.query,
                top_n=args.top_chapters,
                pool=args.route_pool,
            )
            chapters = [chapter for chapter, _ in routed]
            if not chapters:
                raise ValueError(
                    "Could not identify a chapter from the initial dense search. "
                    "Try a larger --route-pool or run without --route."
                )

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
        print("Routed chapters: " + ", ".join(f"{chapter} ({score:.3f})" for chapter, score in routed))
    else:
        print(f"Chapter filter: {chapters or 'None (global search)'}")
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
