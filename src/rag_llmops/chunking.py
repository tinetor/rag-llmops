import re

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


ARTICLE_RE = re.compile(
    r"(?m)^[ \t]*Article[ \t]+(\d+)[ \t]*\r?$"
)

ANNEX_RE = re.compile(
    r"(?mi)^[ \t]*ANNEX[ \t]+[IVXLCDM]+[ \t]*$"
)

# "CHAPTER II" on one line, the chapter title on the next
# (blank lines between them are tolerated).
CHAPTER_RE = re.compile(
    r"(?m)^[ \t]*CHAPTER[ \t]+([IVXLCDM]+)[ \t]*\r?\n"
    r"(?:[ \t]*\r?\n)*"
    r"[ \t]*([^\r\n]+?)[ \t]*\r?$"
)


def _build_full_text(raw_docs: list) -> tuple[str, list[dict]]:
    """Join page texts and track each page's global character offsets."""
    parts = []
    page_ranges = []
    offset = 0

    for doc in raw_docs:
        page_text = doc.text or ""

        if parts:
            parts.append("\n")
            offset += 1

        start = offset
        parts.append(page_text)
        offset += len(page_text)

        page_ranges.append({
            "start": start,
            "end": offset,
            "metadata": dict(doc.metadata),
        })

    return "".join(parts), page_ranges


def _article_matches(text: str):
    """Find article headings and identify where annexes begin."""
    annex_match = ANNEX_RE.search(text)
    article_text = text[:annex_match.start()] if annex_match else text

    return article_text, list(ARTICLE_RE.finditer(article_text)), (
        annex_match.start() if annex_match else len(text)
    )


def _chapter_matches(text: str, article_text_end: int) -> list:
    """Find chapter headings (only before the annexes)."""
    return list(CHAPTER_RE.finditer(text[:article_text_end]))


def _metadata_for_range(
    start: int,
    end: int,
    page_ranges: list[dict],
    article_matches: list,
    chapter_matches: list,
    article_text_end: int,
) -> dict:
    """Map a chunk's global character range to pages, chapter and article."""
    overlapping = [
        page for page in page_ranges
        if page["start"] < end and page["end"] > start
    ]

    metadata = dict(overlapping[0]["metadata"]) if overlapping else {}

    if overlapping:
        first_page = overlapping[0]["metadata"].get("page_label")
        last_page = overlapping[-1]["metadata"].get("page_label")

        metadata["page_label"] = first_page
        metadata["page_start"] = first_page
        metadata["page_end"] = last_page

    # Chunks beginning in the annexes must not inherit Article 113
    # nor the last chapter.
    if start >= article_text_end:
        metadata["article"] = None
        metadata["chapter"] = None
        metadata["chapter_title"] = None
    else:
        preceding_articles = [
            match for match in article_matches
            if match.start() <= start
        ]
        metadata["article"] = (
            int(preceding_articles[-1].group(1))
            if preceding_articles else None
        )

        preceding_chapters = [
            match for match in chapter_matches
            if match.start() <= start
        ]
        if preceding_chapters:
            metadata["chapter"] = preceding_chapters[-1].group(1)
            metadata["chapter_title"] = preceding_chapters[-1].group(2)
        else:
            metadata["chapter"] = None
            metadata["chapter_title"] = None

    return metadata


def _split_by_sentence(
    raw_docs: list,
    chunk_size: int,
    overlap_ratio: float,
) -> list:
    """Split each source page, preserving its original page boundaries."""
    full_text, page_ranges = _build_full_text(raw_docs)
    _, article_matches, article_text_end = _article_matches(full_text)
    chapter_matches = _chapter_matches(full_text, article_text_end)

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * overlap_ratio),
    )

    nodes = []

    for page, page_range in zip(raw_docs, page_ranges):
        page_nodes = splitter.get_nodes_from_documents([page])

        for node in page_nodes:
            local_start = (
                node.start_char_idx
                if node.start_char_idx is not None
                else 0
            )
            local_end = (
                node.end_char_idx
                if node.end_char_idx is not None
                else len(node.text)
            )

            global_start = page_range["start"] + local_start
            global_end = page_range["start"] + local_end

            metadata = _metadata_for_range(
                global_start,
                global_end,
                page_ranges,
                article_matches,
                chapter_matches,
                article_text_end,
            )

            node.metadata.update(metadata)
            node.excluded_embed_metadata_keys = list(node.metadata.keys())
            nodes.append(node)

    return nodes


def _split_by_article(
    raw_docs: list,
    chunk_size: int,
    overlap_ratio: float,
) -> list:
    """Split the preamble and each article into separate segments."""
    full_text, page_ranges = _build_full_text(raw_docs)
    article_text, article_matches, article_text_end = _article_matches(
        full_text
    )
    chapter_matches = _chapter_matches(full_text, article_text_end)

    if not article_matches:
        raise ValueError("No article headings detected. Check ARTICLE_RE.")

    segments = []
    first_start = article_matches[0].start()

    if article_text[:first_start].strip():
        segments.append({
            "start": 0,
            "end": first_start,
            "article": None,
        })

    for i, match in enumerate(article_matches):
        start = match.start()
        next_article_start = (
            article_matches[i + 1].start()
            if i + 1 < len(article_matches)
            else len(article_text)
        )

        # A chapter heading sits right before the first article of the
        # chapter; cut the segment there so the heading does not pollute
        # the previous article's chunk.
        chapter_in_between = [
            chapter.start() for chapter in chapter_matches
            if start < chapter.start() < next_article_start
        ]
        end = (
            min(chapter_in_between)
            if chapter_in_between
            else next_article_start
        )

        if article_text[start:end].strip():
            segments.append({
                "start": start,
                "end": end,
                "article": int(match.group(1)),
            })

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * overlap_ratio),
    )

    nodes = []

    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        original_segment = article_text[start:end]
        segment_text = original_segment.strip()

        leading_whitespace = (
            len(original_segment) - len(original_segment.lstrip())
        )
        document_start = start + leading_whitespace

        doc = Document(
            text=segment_text,
            metadata={"article": segment["article"]},
        )
        doc.excluded_embed_metadata_keys = ["article"]

        for node in splitter.get_nodes_from_documents([doc]):
            local_start = (
                node.start_char_idx
                if node.start_char_idx is not None
                else 0
            )
            local_end = (
                node.end_char_idx
                if node.end_char_idx is not None
                else len(node.text)
            )

            global_start = document_start + local_start
            global_end = document_start + local_end

            metadata = _metadata_for_range(
                global_start,
                global_end,
                page_ranges,
                article_matches,
                chapter_matches,
                article_text_end,
            )

            node.metadata.update(metadata)
            node.excluded_embed_metadata_keys = list(node.metadata.keys())
            nodes.append(node)

    return nodes


def chunking(
    raw_text: list,
    max_chars: int = 1500,
    overlap_ratio: float = 0.1,
    strategy: str = "sentence",
) -> list:
    """Split documents using sentence or article-aware chunking."""
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be between 0 and 1.")

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0.")

    if strategy == "sentence":
        return _split_by_sentence(
            raw_text,
            chunk_size=max_chars,
            overlap_ratio=overlap_ratio,
        )

    if strategy == "article":
        return _split_by_article(
            raw_text,
            chunk_size=max_chars,
            overlap_ratio=overlap_ratio,
        )

    raise ValueError("Use 'sentence' or 'article'.")


if __name__ == "__main__":
    from rag_llmops.config import load_config
    from rag_llmops.load_pdf import load_pdf

    config = load_config()
    docs = load_pdf(config["data"]["pdf_path"])

    for strategy in ("sentence", "article"):
        chunks = chunking(
            docs,
            max_chars=config["chunking"]["max_chars"],
            overlap_ratio=config["chunking"]["overlap_ratio"],
            strategy=strategy,
        )

        print(f"\nStrategy: {strategy}")
        print(f"Chunks generated: {len(chunks)}")

        shown = 0
        for i, chunk in enumerate(chunks):
            if chunk.metadata.get("article") is not None:
                print(
                    f"[{i}] chapter={chunk.metadata.get('chapter')} "
                    f"({chunk.metadata.get('chapter_title')}) | "
                    f"article={chunk.metadata.get('article')} | "
                    f"page={chunk.metadata.get('page_label')} | "
                    f"text={chunk.text[:100]!r}"
                )
                shown += 1

                if shown == 5:
                    break
