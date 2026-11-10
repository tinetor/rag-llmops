"""Entry point for document ingestion."""


from rag_llmops.ingestion_pipeline import run_ingestion


def main():
    index, nodes = run_ingestion()
    print(f"Ingestion completed: {len(nodes)} nodes indexed.")


if __name__ == "__main__":
    main()
