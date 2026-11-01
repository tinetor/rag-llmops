import torch
from langfuse import get_client

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
    langfuse = get_client()
    model_name = getattr(model.config, "_name_or_path", "unknown")

    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-pipeline",
        input={"question": question},
    ) as trace:

        # 1. Retrieve relevant chunks
        with langfuse.start_as_current_observation(
            as_type="span",
            name="retrieval",
            input={"question": question},
        ) as retrieval_span:
            passages = retrieve(retriever, question)

            retrieval_span.update(
                output={
                    "chunk_count": len(passages),
                    "chunks": passages,
                }
            )

        # 2. Build the prompt with the retrieved context
        prompt = build_prompt(question, passages)

        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(chat, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = inputs.to(device)

        # 3. Generate the answer with Hugging Face
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="generate-answer",
            model=model_name,
            input={"prompt": chat},
            model_parameters={
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            },
        ) as generation:
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

            generation.update(
                output=answer,
                usage_details={
                    "input": inputs["input_ids"].shape[1],
                    "output": len(new_tokens),
                },
            )

        result = {
            "answer": answer,
            "sources": passages,
            "escaped": ESCAPE_PHRASE in answer,
        }

        # 4. Record the final pipeline output
        trace.update(
            output={
                "answer": result["answer"],
                "escaped": result["escaped"],
            }
        )

    return result


if __name__ == "__main__":
    import argparse
    from config import load_config

    parser = argparse.ArgumentParser(
        description="Run a single RAG question with Qwen."
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask the RAG pipeline.",
    )
    args = parser.parse_args()

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

    question = args.question

    result = generate_answer(
        question=question,
        retriever=retriever,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    # Ensure pending traces are sent before the script exits
    langfuse = get_client()
    langfuse.flush()

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
