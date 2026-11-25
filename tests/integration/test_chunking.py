from sentence_transformers import SentenceTransformer

from rag_llmops.chunking import chunking
from rag_llmops.config import load_config
from rag_llmops.load_pdf import load_pdf


def test_chunks_fit_embedding_window():
    config = load_config()
    model = SentenceTransformer(config["embedding"]["model_name"])
    docs = load_pdf(config["data"]["pdf_path"])
    chunks = chunking(
        docs,
        chunk_size=config["chunking"]["chunk_size"],
        overlap_ratio=config["chunking"]["overlap_ratio"],
    )
    sizes = [len(model.tokenizer(c.text)["input_ids"]) for c in chunks]
    too_long = [s for s in sizes if s > model.max_seq_length]
    assert not too_long, (
        f"{len(too_long)}/{len(chunks)} chunks exceed {model.max_seq_length} tokens (worst: {max(sizes)})"
    )
