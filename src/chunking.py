# Add near the top with the other regexes

CHAPTER_RE = re.compile(
    r"(?m)^[ \t]*CHAPTER[ \t]+([IVXLCDM]+)[ \t]*\r?$"
)


# Replace the entire _article_matches function with this version

def _article_matches(text: str):
    """Find article and chapter headings and identify where annexes begin."""
    annex_match = ANNEX_RE.search(text)
    article_text = text[:annex_match.start()] if annex_match else text

    return (
        article_text,
        list(ARTICLE_RE.finditer(article_text)),
        list(CHAPTER_RE.finditer(article_text)),
        annex_match.start() if annex_match else len(text),
    )


# Replace the entire _metadata_for_range function with this version

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

    if start >= article_text_end:
        metadata["article"] = None
        metadata["chapter"] = None
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

        metadata["chapter"] = (
            preceding_chapters[-1].group(1)
            if preceding_chapters else None
        )

    return metadata


# In _split_by_sentence replace

_, article_matches, article_text_end = _article_matches(full_text)

# with

_, article_matches, chapter_matches, article_text_end = (
    _article_matches(full_text)
)

# and replace

metadata = _metadata_for_range(
    global_start,
    global_end,
    page_ranges,
    article_matches,
    article_text_end,
)

# with

metadata = _metadata_for_range(
    global_start,
    global_end,
    page_ranges,
    article_matches,
    chapter_matches,
    article_text_end,
)


# In _split_by_article replace

article_text, article_matches, article_text_end = _article_matches(
    full_text
)

# with

article_text, article_matches, chapter_matches, article_text_end = (
    _article_matches(full_text)
)

# and replace

metadata = _metadata_for_range(
    global_start,
    global_end,
    page_ranges,
    article_matches,
    article_text_end,
)

# with

metadata = _metadata_for_range(
    global_start,
    global_end,
    page_ranges,
    article_matches,
    chapter_matches,
    article_text_end,
)
