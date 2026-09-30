"""Loads a saved index + chunks and retrieves the top-k chunks for a query."""

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from chunking import load_config


def load_retriever(config):
    model = SentenceTransformer(config["embedding"]["model_name"])
    index = faiss.read_index("rag_index.faiss")
    chunks = json.loads(Path("chunks.json").read_text())
    return model, index, chunks


def retrieve(query, model, index, chunks, k=5):
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, k)
    return [{**chunks[idx], "score": float(score)} for score, idx in zip(scores[0], indices[0])]


if __name__ == "__main__":
    config = load_config()
    model, index, chunks = load_retriever(config)

    query = "can I accept a gift from a supplier?"
    results = retrieve(query, model, index, chunks, k=config["retrieval"]["k"])

    print(f"Query: {query}\n")
    for r in results:
        print(f"[{r['score']:.3f}] {r['heading']}")
