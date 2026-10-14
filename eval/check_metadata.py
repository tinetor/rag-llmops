
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from chunking import chunking
from load_pdf import load_pdf


def validate_metadata(chunks):
    errors = []

    for i, chunk in enumerate(chunks):
        metadata = chunk.metadata
        text = chunk.text.strip()

        # Verifica se o texto não está vazio
        if not text:
            errors.append((i, "texto vazio"))

        # Verifica se existe a página inicial
        if not metadata.get("page_label"):
            errors.append((i, "page_label ausente"))

        # Verifica se o intervalo de páginas é válido
        start = metadata.get("page_start")
        end = metadata.get("page_end")

        try:
            if start is not None and end is not None:
                if int(start) > int(end):
                    errors.append(
                        (i, f"intervalo de páginas inválido: {start}-{end}")
                    )
        except (TypeError, ValueError):
            errors.append((i, "metadados de página não numéricos"))

        # Verifica se o artigo, quando definido, é válido
        article = metadata.get("article")

        if article is not None:
            try:
                article_number = int(article)
                if not 1 <= article_number <= 113:
                    errors.append(
                        (i, f"número de artigo inválido: {article}")
                    )
            except (TypeError, ValueError):
                errors.append((i, f"artigo não numérico: {article}"))

    print(f"Chunks verificados: {len(chunks)}")
    print(f"Problemas encontrados: {len(errors)}")

    if errors:
        print("\nPrimeiros problemas encontrados:")
        for index, message in errors[:10]:
            print(f"  Chunk {index}: {message}")
    else:
        print("Todas as verificações estruturais passaram.")


def show_article_examples(chunks, limit=5):
    print(f"\nExemplos de chunks com artigo atribuído (máximo {limit}):")

    shown = 0

    for i, chunk in enumerate(chunks):
        article = chunk.metadata.get("article")

        if article is not None:
            print(f"\nChunk {i}")
            print(f"Artigo: {article}")
            print(f"Página inicial: {chunk.metadata.get('page_start')}")
            print(f"Página final: {chunk.metadata.get('page_end')}")
            print(f"Texto: {chunk.text[:160]!r}")

            shown += 1

            if shown >= limit:
                break

    if shown == 0:
        print("Nenhum chunk com artigo atribuído.")


def main():
    print("CHECK_METADATA: script iniciado", flush=True)

    config = load_config()

    print("A carregar PDF...", flush=True)
    docs = load_pdf(config["data"]["pdf_path"])
    print(f"Documentos carregados: {len(docs)}", flush=True)

    for strategy in ("sentence", "article"):
        print("\n" + "=" * 60)
        print(f"ESTRATÉGIA: {strategy}")

        chunks = chunking(
            docs,
            max_chars=config["chunking"]["max_chars"],
            overlap_ratio=config["chunking"]["overlap_ratio"],
            strategy=strategy,
        )

        print(f"Total de chunks: {len(chunks)}")

        print("\n--- Validação dos metadados ---")
        validate_metadata(chunks)

        print("\n--- Exemplos ---")
        show_article_examples(chunks)


if __name__ == "__main__":
    main()
