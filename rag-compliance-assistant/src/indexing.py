"""Builds and persists a FAISS index from chunks."""

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from chunking import load_config, chunk_by_heading


def build_index(chunks, model_name):
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


if __name__ == "__main__":
    config = load_config()
    raw_text = Path("data/sample_corpus.txt").read_text()
    chunks = chunk_by_heading(
        raw_text,
        min_chars=config["chunking"]["min_chars"],
        max_chars=config["chunking"]["max_chars"],
    )
    index = build_index(chunks, config["embedding"]["model_name"])

    faiss.write_index(index, "rag_index.faiss")
    Path("chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2))

    print(f"Index built: {index.ntotal} chunks, dim {index.d}")
    print("Saved rag_index.faiss + chunks.json")
