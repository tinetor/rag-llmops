
import sys
from pathlib import Path

import yaml
from langfuse import Evaluation, get_client

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from generation import generate_answer
from llm import load_llm
from retrieval import load_retriever


def main():
    config_path = ROOT_DIR / "configs" / "config.yaml"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    retriever = load_retriever(
        persist_dir=config["index"]["persist_dir"],
        model_name=config["embedding"]["model_name"],
        k=config["retrieval"]["top_k"],
        mode=config["retrieval"]["mode"],
        candidate_k=config["retrieval"]["candidate_k"],
    )

    model, tokenizer = load_llm(config)

    langfuse = get_client()
    dataset = langfuse.get_dataset("Golden_Dataset")

    def generation_task(*, item, **kwargs):
        question = item.input["question"]

        result = generate_answer(
            question=question,
            retriever=retriever,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config["generation"]["max_new_tokens"],
        )

        retrieved_articles = sorted({
            str(source["article"])
            for source in result["sources"]
            if source.get("article") is not None
        })

        return {
            "question": question,
            "answer": result["answer"],
            "retrieved_articles": retrieved_articles,
            "sources": result["sources"],
            "escaped": result["escaped"],
        }

    def article_hit_evaluator(*, output, expected_output, **kwargs):
        expected = set(map(str, expected_output["articles"]))
        retrieved = set(output["retrieved_articles"])
        hit = bool(expected & retrieved)

        return Evaluation(
            name="article_hit",
            value=1.0 if hit else 0.0,
            comment=(
                f"Expected: {sorted(expected)}; "
                f"Retrieved: {sorted(retrieved)}"
            ),
        )

    dataset.run_experiment(
        name="Generation evaluation",
        task=generation_task,
        evaluators=[article_hit_evaluator],
    )


if __name__ == "__main__":
    main()
