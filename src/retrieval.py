
import json

from qdrant_client import QdrantClient
from llama_index.core import Settings, VectorStoreIndex
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


def load_retriever(
    persist_dir: str,
    collection_name: str,
    model_name: str,
    k: int,
    mode: str,
    candidate_k: int,
    chapter: str | None = None,
):
    """
    Load a Qdrant retriever with an optional chapter filter.

    Args:
        persist_dir: Local Qdrant storage directory.
        collection_name: Qdrant collection name.
        model_name: Hugging Face embedding model name.
        k: Number of final results.
        mode: Retrieval mode, either "dense" or "hybrid".
        candidate_k: Candidate pool size for hybrid retrieval.
        chapter: Optional chapter identifier, such as "III".

    Returns:
        A configured LlamaIndex retriever.
    """
    if mode not in ("dense", "hybrid"):
        raise ValueError("mode must be either 'dense' or 'hybrid'.")

    embed_model = HuggingFaceEmbedding(model_name=model_name)

    client = QdrantClient(path=persist_dir)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    metadata_filters = None

    if chapter is not None:
        metadata_filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="chapter",
                    value=chapter,
                    operator=FilterOperator.EQ,
                )
            ]
        )

    if mode == "dense":
        return index.as_retriever(
            similarity_top_k=k,
            filters=metadata_filters,
        )

    pool = max(candidate_k, k)

    nodes = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            if chapter is not None and payload.get("chapter") != chapter:
                continue

            node_data = payload.get("_node_content")

            if not node_data:
                continue

            if isinstance(node_data, str):
                node_data = json.loads(node_data)

            nodes.append(TextNode.from_dict(node_data))

        if offset is None:
            break

    if not nodes:
        if chapter is not None:
            raise ValueError(
                f"No nodes found for chapter {chapter!r}."
            )

        raise ValueError(
            "No text nodes were found in the Qdrant collection."
        )

    vector_retriever = index.as_retriever(
        similarity_top_k=pool,
        filters=metadata_filters,
    )

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=pool,
    )

    Settings.llm = None

    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )


def retrieve(query: str, retriever):
    """
    Retrieve relevant chunks and return their text and metadata.
    """
    results = retriever.retrieve(query)

    output = []

    for result in results:
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

if __name__ == "__main__":
    import argparse
    from config import load_config

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
        type=str,
        default=None,
        help="Optional chapter filter, e.g. III.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to return.",
    )
    args = parser.parse_args()

    config = load_config()

    mode = args.mode or config["retrieval"]["mode"]
    top_k = args.top_k or config["retrieval"]["top_k"]

    retriever = load_retriever(
        persist_dir=config["index"]["persist_dir"] + "_qdrant",
        collection_name=config["index"]["collection_name"],
        model_name=config["embedding"]["model_name"],
        k=top_k,
        mode=mode,
        candidate_k=config["retrieval"]["candidate_k"],
        chapter=args.chapter,
    )

    results = retrieve(args.query, retriever)

    print(f"\nQuery: {args.query}")
    print(f"Mode: {mode}")
    print(f"Chapter filter: {args.chapter or 'None'}")
    print(f"Results: {len(results)}")

    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Chapter: {result['chapter']}")
        print(f"Article: {result['article']}")
        print(f"Chapter title: {result['chapter_title']}")
        print(f"Score: {result['score']}")
        print(result["text"][:500])