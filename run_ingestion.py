"""Entry point for document ingestion."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ingestion_pipeline import run_ingestion


def main():
    index, nodes = run_ingestion()
    print(f"Ingestion completed: {len(nodes)} nodes indexed.")


if __name__ == "__main__":
    main()
