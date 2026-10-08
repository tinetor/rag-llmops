from llama_index.core.node_parser import SentenceSplitter


def chunking(raw_text: list, max_chars: int = 1500, overlap_ratio: float = 0.1) -> list:
    # raw_text is the list of Documents from load_pdf.
    # max_chars is passed to SentenceSplitter as chunk_size, which counts tokens, not characters.
    for doc in raw_text:
        doc.excluded_embed_metadata_keys = list(doc.metadata.keys())

    splitter = SentenceSplitter(
        chunk_size=max_chars,
        chunk_overlap=int(max_chars * overlap_ratio),
    )
    nodes = splitter.get_nodes_from_documents(raw_text)
    return nodes


if __name__ == "__main__":
    from config import load_config
    from load_pdf import load_pdf

    config = load_config()
    docs = load_pdf(config["data"]["pdf_path"])
    chunks = chunking(
        docs,
        max_chars=config["chunking"]["max_chars"],
        overlap_ratio=config["chunking"]["overlap_ratio"],
    )

    print(f"{len(chunks)} chunks generated from {config['data']['pdf_path']}")
    print(f"longest chunk: {max(len(c.text) for c in chunks)} chars")
    for i, c in enumerate(chunks[:5]):
        print(f"[{i}] page {c.metadata.get('page_label')} | {len(c.text)} chars | {c.text[:80]!r}")
