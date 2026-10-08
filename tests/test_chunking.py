from sentence_transformers import SentenceTransformer

from chunking import chunking
from config import load_config
from load_pdf import load_pdf

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def test_chunks_fit_embedding_window():
    config = load_config()
    model = SentenceTransformer(MODEL_NAME)
    docs = load_pdf(config["data"]["pdf_path"])
    chunks = chunking(docs, max_chars=config["chunking"]["max_chars"])
    sizes = [len(model.tokenizer(c.text)["input_ids"]) for c in chunks]
    too_long = [s for s in sizes if s > model.max_seq_length]
    assert not too_long, (
        f"{len(too_long)}/{len(chunks)} chunks exceed {model.max_seq_length} tokens (worst: {max(sizes)})"
    )
