from sentence_transformers import SentenceTransformer

from chunking import chunking
from load_pdf import load_pdf

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def test_chunks_fit_embedding_window():
    model = SentenceTransformer(MODEL_NAME)
    chunks = chunking(load_pdf("data/ai_act.pdf"))
    sizes = [len(model.tokenizer(c.text)["input_ids"]) for c in chunks]
    too_long = [s for s in sizes if s > model.max_seq_length]
    assert not too_long, (
        f"{len(too_long)}/{len(chunks)} chunks exceed {model.max_seq_length} tokens (worst: {max(sizes)})"
    )
