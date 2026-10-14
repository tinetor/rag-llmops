
import re
from copy import copy

from config import load_config
from load_pdf import load_pdf


# Match article headings that appear on a line by themselves,
# such as "Article 5" or "Article 113".
ARTICLE_RE = re.compile(
    r"(?m)^[ \t]*Article[ \t]+(\d+)[ \t]*\r?$"
)

# Match annex headings such as "ANNEX I" or "ANNEX II".
ANNEX_RE = re.compile(
    r"(?mi)^[ \t]*ANNEX[ \t]+[IVXLCDM]+[ \t]*$"
)


def split_into_articles(text: str) -> list[dict]:
    """
    Split the main document text into article blocks.

    Each block contains:
    - article: the article number
    - start: the start offset in the source text
    - end: the end offset in the source text
    - text: the complete text between this heading
      and the next detected article heading

    Text starting at the first detected annex heading is excluded.
    """
    # Exclude annexes to reduce false positives from article references.
    annex_match = ANNEX_RE.search(text)

    if annex_match:
        text = text[:annex_match.start()]

    # Find all article headings in the remaining text.
    matches = list(ARTICLE_RE.finditer(text))
    articles = []

    for i, match in enumerate(matches):
        start = match.start()

        # The current article ends where the next article begins.
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        article_text = text[start:end].strip()

        articles.append({
            "article": int(match.group(1)),
            "start": start,
            "end": end,
            "text": article_text,
        })

    return articles


def inspect_articles(docs) -> list[dict]:
    """
    Extract article blocks and generate diagnostic information.

    Title detection is heuristic and should be reviewed manually.
    """
    # Join the extracted PDF pages into one text string.
    full_text = "\n".join(doc.text or "" for doc in docs)

    articles = split_into_articles(full_text)
    results = []

    for article in articles:
        # Keep non-empty lines for easier inspection.
        lines = [
            line.strip()
            for line in article["text"].splitlines()
            if line.strip()
        ]

        # The line immediately after "Article N" is assumed
        # to be the title. This is only a diagnostic heuristic.
        title = lines[1] if len(lines) > 1 else ""
        title_words = len(title.split())

        # Check whether a line starts with paragraph "1.".
        # Some articles may legitimately have a different structure.
        has_paragraph_1 = any(
            re.match(r"^1\.", line)
            for line in lines[2:]
        )

        results.append({
            "article": article["article"],
            "start": article["start"],
            "end": article["end"],
            "length": len(article["text"]),
            "title": title,
            "title_words": title_words,
            "under_8_words": title_words <= 8,
            "has_paragraph_1": has_paragraph_1,
            "paragraph_preview": (
                lines[2] if len(lines) > 2 else ""
            ),
            "text": article["text"],
        })

    return results


def validate_article_boundaries(results: list[dict]) -> list[dict]:
    """
    Perform basic checks on the detected article boundaries.

    Checks:
    1. Each block starts with its expected article heading.
    2. The next detected article heading does not appear as
       a standalone heading inside the current block.

    These checks do not prove that every heading was detected.
    Manual inspection is still required.
    """
    problems = []

    for i, item in enumerate(results):
        article_number = item["article"]
        article_text = item["text"]

        # Remove blank lines before checking the first line.
        lines = [
            line.strip()
            for line in article_text.splitlines()
            if line.strip()
        ]

        # Confirm that the block starts with its expected heading.
        starts_correctly = bool(
            lines
            and re.fullmatch(
                rf"Article\s+{article_number}",
                lines[0],
                flags=re.IGNORECASE,
            )
        )

        # Identify the next article number in the detected sequence.
        next_article = (
            results[i + 1]["article"]
            if i + 1 < len(results)
            else None
        )

        contains_next_heading = False

        if next_article is not None:
            next_heading_pattern = re.compile(
                rf"(?m)^[ \t]*Article[ \t]+"
                rf"{next_article}[ \t]*\r?$",
                flags=re.IGNORECASE,
            )

            contains_next_heading = bool(
                next_heading_pattern.search(article_text)
            )

        # Report only blocks that fail one of the checks.
        if not starts_correctly or contains_next_heading:
            problems.append({
                "article": article_number,
                "starts_correctly": starts_correctly,
                "contains_next_heading": contains_next_heading,
                "first_lines": lines[:3],
                "last_lines": lines[-3:],
            })

    return problems


def main():
    """Load the PDF, inspect article blocks, and print diagnostics."""
    config = load_config()
    docs = load_pdf(config["data"]["pdf_path"])

    # Find the first annex heading across the extracted PDF pages.
    docs_before_annex = []
    annex_found = False

    for doc in docs:
        text = doc.text or ""
        match = ANNEX_RE.search(text)

        if match:
            # Keep the text before the annex heading on this page.
            text_before_annex = text[:match.start()]

            if text_before_annex.strip():
                # Preserve the original document and its metadata.
                doc_copy = copy(doc)
                doc_copy.text = text_before_annex
                docs_before_annex.append(doc_copy)

            annex_found = True
            break

        docs_before_annex.append(doc)

    if not annex_found:
        print(
            "WARNING: No annex heading was found. "
            "The inspection may include annex content."
        )

    # Detect articles and collect diagnostic information.
    results = inspect_articles(docs_before_annex)

    print("\n" + "=" * 70)
    print("ARTICLE STRUCTURE REPORT")
    print("=" * 70)
    print(f"Documents loaded: {len(docs)}")
    print(f"Documents inspected: {len(docs_before_annex)}")
    print(f"Annex boundary found: {annex_found}")
    print(f"Articles detected: {len(results)}")

    # Print the detected article numbers and their probable titles.
    print("\n--- DETECTED ARTICLES ---")

    for item in results:
        print(
            f"Article {item['article']} | "
            f"{item['length']} characters | "
            f"Probable title: {item['title']}"
        )

    # Check for duplicate and missing article numbers.
    print("\n--- ARTICLE NUMBER SEQUENCE ---")

    numbers = [item["article"] for item in results]
    duplicates = sorted({
        number
        for number in numbers
        if numbers.count(number) > 1
    })

    print(
        f"First article: "
        f"{numbers[0] if numbers else 'None'}"
    )
    print(
        f"Last article: "
        f"{numbers[-1] if numbers else 'None'}"
    )
    print(f"Duplicate numbers: {duplicates}")

    if numbers:
        expected = set(range(1, max(numbers) + 1))
        found = set(numbers)

        print(
            "Missing numbers within the detected range: "
            f"{sorted(expected - found)}"
        )
    else:
        print("No article numbers were detected.")

    # Check whether article blocks pass the basic boundary tests.
    boundary_problems = validate_article_boundaries(results)

    print("\n--- ARTICLE BOUNDARY VALIDATION ---")
    print(f"Articles checked: {len(results)}")
    print(f"Blocks with potential problems: {len(boundary_problems)}")

    for problem in boundary_problems:
        print(f"\nArticle {problem['article']}")
        print(
            f"Starts correctly: "
            f"{problem['starts_correctly']}"
        )
        print(
            f"Contains next heading: "
            f"{problem['contains_next_heading']}"
        )
        print(f"First lines: {problem['first_lines']}")
        print(f"Last lines: {problem['last_lines']}")

    # Flag long probable titles for manual inspection.
    long_titles = [
        item for item in results
        if not item["under_8_words"]
    ]

    print("\n--- TITLES WITH MORE THAN 8 WORDS ---")

    if long_titles:
        for item in long_titles:
            print(
                f"Article {item['article']} | "
                f"{item['title']} "
                f"({item['title_words']} words)"
            )
    else:
        print("None.")

    # Flag articles without an apparent paragraph "1.".
    missing_paragraph = [
        item for item in results
        if not item["has_paragraph_1"]
    ]

    print("\n--- ARTICLES TO REVIEW ---")

    if missing_paragraph:
        for item in missing_paragraph:
            print(
                f"Article {item['article']} | "
                f"Title: {item['title']} | "
                f"Next line: {item['paragraph_preview']!r}"
            )
    else:
        print("None.")

    # Print samples from the first three blocks.
    # Reviewing both ends helps reveal suspicious boundaries.
    print("\n--- ARTICLE BLOCK SAMPLES ---")

    for item in results[:3]:
        print(f"\nArticle {item['article']}")
        print("START:")
        print(item["text"][:500])
        print("\nEND:")
        print(item["text"][-300:])


    # Print all detected article blocks.
    print("\n" + "=" * 70)
    print("FULL TEXT OF ALL DETECTED ARTICLES")
    print("=" * 70)

    for item in results:
        print(f"\n{'=' * 30}")
        print(f"ARTICLE {item['article']}")
        print(f"Characters: {len(item['text'])}")
        print(f"{'=' * 30}")

        # Print the complete text of the current article.
        print(item["text"])

        print(f"\n--- END OF ARTICLE {item['article']} ---")

    print("\n" + "=" * 70)
    print(f"Total articles printed: {len(results)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
