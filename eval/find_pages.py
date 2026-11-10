"""List the pages where a term appears, using the same page labels the retriever reports.

Usage (from the repo root):
    python3 eval/find_pages.py "Article 5" --heading
    python3 eval/find_pages.py "serious incident"
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from rag_llmops.config import load_config
from rag_llmops.load_pdf import load_pdf


def main():
    args = [a for a in sys.argv[1:] if a != "--heading"]
    heading_only = "--heading" in sys.argv[1:]
    if not args:
        sys.exit('usage: python3 eval/find_pages.py "term" [--heading]')
    term = args[0]

    pattern = re.escape(term)
    if term[-1].isdigit():
        pattern += r"(?!\d)"  # "Article 5" must not match "Article 50"
    if heading_only:
        pattern = r"^" + pattern  # only where a line starts with the term
    regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

    docs = load_pdf(str(ROOT / load_config()["data"]["pdf_path"]))
    found = 0
    for doc in docs:
        m = regex.search(doc.text)
        if m:
            found += 1
            start = max(0, m.start() - 40)
            snippet = doc.text[start:m.end() + 80].replace("\n", " ")
            print(f"page {doc.metadata.get('page_label')}: ...{snippet}...")
    print(f"\n{found} page(s) matched")


if __name__ == "__main__":
    main()
