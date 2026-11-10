"""Tests that PDF extraction produced clean, readable text (works for any PDF)."""

import pytest

from rag_llmops.load_pdf import load_pdf

PDF_PATH = "data/ai_act.pdf"
MIN_CHARS_PER_PAGE = 100
MIN_AVG_WORD_LEN = 3
MAX_AVG_WORD_LEN = 12


@pytest.fixture(scope="module")
def docs():
    return load_pdf(PDF_PATH)


def test_no_replacement_characters(docs):
    for i, doc in enumerate(docs):
        assert "\ufffd" not in doc.text, f"page {i} has a replacement character (broken font mapping)"


def test_word_lengths_look_reasonable(docs):
    for i, doc in enumerate(docs):
        words = doc.text.split()
        if not words:
            continue
        avg_len = sum(len(w) for w in words) / len(words)
        assert MIN_AVG_WORD_LEN < avg_len < MAX_AVG_WORD_LEN, (
            f"page {i}: average word length {avg_len:.1f} looks wrong "
            f"(words glued together or falling apart)"
        )


def test_pages_have_minimum_content(docs):
    empty_pages = [i for i, doc in enumerate(docs) if len(doc.text.strip()) < MIN_CHARS_PER_PAGE]
    assert len(empty_pages) < len(docs) * 0.1, f"too many near-empty pages: {empty_pages}"