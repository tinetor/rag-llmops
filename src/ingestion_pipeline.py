
from config import load_config
from load_pdf import load_pdf
from chunking import chunking
from indexing import build_index


def run_ingestion_pipeline(config=None):

    # 1. Load configuration
    if config is None:
        config = load_config()

    # 2. Load PDF documents
    docs = load_pdf(config["data"]["pdf_path"])

    # 3. Split documents into nodes
    nodes = chunking(
        docs,
        max_chars=config["chunking"]["max_chars"],
        overlap_ratio=config["chunking"]["overlap_ratio"],
        strategy=config["chunking"]["strategy"],
    )

    # 4. Build Qdrant index
    index = build_index(
        nodes,
        model_name=config["embedding"]["model_name"],
        persist_dir=config["index"]["persist_dir"] + "_qdrant",
        collection_name=config["index"]["collection_name"],
    )

    return {
        "documents": len(docs),
        "nodes": len(nodes),
        "index": index,
    }


if __name__ == "__main__":
    result = run_ingestion_pipeline()

    print("Ingestion completed successfully!")
    print(f"Documents loaded: {result['documents']}")
    print(f"Nodes indexed: {result['nodes']}")

