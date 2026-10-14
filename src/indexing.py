



if __name__ == "__main__":
    from chunking import chunking
    from config import load_config
    from load_pdf import load_pdf

    config = load_config()
    docs = load_pdf(config["data"]["pdf_path"])

    nodes = chunking(
        docs,
        max_chars=config["chunking"]["max_chars"],
        overlap_ratio=config["chunking"]["overlap_ratio"],
        strategy=config["chunking"]["strategy"],
    )


    index = build_index(nodes, model_name=config["embedding"]["model_name"])
    index.storage_context.persist(persist_dir=config["index"]["persist_dir"])
    print(f"Index built and persisted: {len(nodes)} nodes")
