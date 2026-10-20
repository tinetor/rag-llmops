
import torch

from llm import load_llm
from prompts import ESCAPE_PHRASE, build_prompt
from retrieval import load_retriever, retrieve


def generate_answer(
    question: str,
    retriever,
    model,
    tokenizer,
    max_new_tokens: int,
) -> dict:
    passages = retrieve(retriever, question)
    prompt = build_prompt(question, passages)

    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        chat,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return {
        "answer": answer,
        "sources": passages,
        "escaped": ESCAPE_PHRASE in answer,
    }


if __name__ == "__main__":
    from config import load_config

    config = load_config()

    retriever = load_retriever(
        persist_dir=config["index"]["persist_dir"],
        model_name=config["embedding"]["model_name"],
        k=config["retrieval"]["top_k"],
        mode=config["retrieval"]["mode"],
        candidate_k=config["retrieval"]["candidate_k"],
    )

    model, tokenizer = load_llm(config)

    max_new_tokens = config["generation"]["max_new_tokens"]

    question = "What are the prohibited AI practices?"

    result = generate_answer(
        question=question,
        retriever=retriever,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    print(f"Q: {question}\n")
    print(f"A: {result['answer']}\n")
    print(f"Escaped: {result['escaped']}")
    print("Sources:")

    for source in result["sources"]:
        score = (
            f"{source['score']:.3f}"
            if source["score"] is not None
            else "N/A"
        )

        article = source["article"]
        article_label = (
            f"Article {article}"
            if article is not None
            else "Article unknown"
        )

        print(f"  [{score}] {article_label}")
        print(f"    {source['text'][:120]!r}")
