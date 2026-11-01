
import argparse
import torch
from langfuse import get_client

from llm import load_llm
from prompts import ESCAPE_PHRASE, build_prompt
from retrieval import (
    load_retriever,
    retrieve,
    load_index,
    route_chapters,
    build_retriever,
)


def generate_answer(
    question: str,
    retriever,
    model,
    tokenizer,
    max_new_tokens: int,
    use_router: bool = False,
    index=None,
    persist_dir: str = None,
    collection_name: str = None,
    embedding_model_name: str = None,
    top_chapters: int = 2,
    route_pool: int = 30,
    retrieval_mode: str = "dense",
    candidate_k: int = 20,
) -> dict:
    langfuse = get_client()
    model_name = getattr(model.config, "_name_or_path", "unknown")

    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-pipeline",
        input={"question": question},
    ) as trace:

        # 1. Optionally route to the most relevant chapters
        if use_router:
            if index is None:
                raise ValueError(
                    "An index is required when use_router=True."
                )

            with langfuse.start_as_current_observation(
                as_type="span",
                name="chapter-routing",
                input={"question": question},
            ) as routing_span:
                routed = route_chapters(
                    index=index,
                    query=question,
                    top_n=top_chapters,
                    pool=route_pool,
                )
                chapters = [chapter for chapter, _ in routed]

                routing_span.update(
                    output={
                        "selected_chapters": chapters,
                        "routing_results": [
                            {"chapter": chapter, "score": score}
                            for chapter, score in routed
                        ],
                    }
                )

            retriever_kwargs = {
                "index": index,
                "persist_dir": persist_dir,
                "collection_name": collection_name,
                "k": retriever.k if hasattr(retriever, "k") else None,
                "mode": retrieval_mode,
                "candidate_k": candidate_k,
            }

            # Avoid passing k=None if the retriever does not expose k.
            if retriever_kwargs["k"] is None:
                retriever_kwargs["k"] = 5

            if chapters:
                retriever_kwargs["chapter"] = chapters

            question_retriever = build_retriever(**retriever_kwargs)
        else:
            question_retriever = retriever

        # 2. Retrieve relevant chunks
        with langfuse.start_as_current_observation(
            as_type="span",
            name="retrieval",
            input={
                "question": question,
                "use_router": use_router,
            },
        ) as retrieval_span:
            passages = retrieve(question_retriever, question)

            retrieval_span.update(
                output={
                    "chunk_count": len(passages),
                    "chunks": passages,
                }
            )

        # 3. Build the prompt with the retrieved context
        prompt = build_prompt(question, passages)

        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(chat, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = inputs.to(device)

        # 4. Generate the answer with Hugging Face
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

        # 5. Record the final pipeline output
        trace.update(
            output={
                "answer": result["answer"],
                "escaped": result["escaped"],
            }
        )

    return result


if __name__ == "__main__":
    from config import load_config

    parser = argparse.ArgumentParser(
        description="Run the RAG pipeline with a custom question."
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask the RAG pipeline.",
    )
    args = parser.parse_args()
    question = args.question

    config = load_config()
    retrieval_config = config["retrieval"]

    persist_dir = config["index"]["persist_dir"] + "_qdrant"
    collection_name = config["index"]["collection_name"]
    embedding_model_name = config["embedding"]["model_name"]

    use_router = retrieval_config.get("use_router", False)
    top_k = retrieval_config["top_k"]
    candidate_k = retrieval_config["candidate_k"]
    mode = retrieval_config["mode"]
    top_chapters = retrieval_config.get("top_chapters", 2)
    route_pool = retrieval_config.get("route_pool", 30)

    if use_router:
        index = load_index(
            persist_dir=persist_dir,
            collection_name=collection_name,
            model_name=embedding_model_name,
        )
        retriever = None
    else:
        index = None
        retriever = load_retriever(
            persist_dir=persist_dir,
            collection_name=collection_name,
            model_name=embedding_model_name,
            k=top_k,
            mode=mode,
            candidate_k=candidate_k,
        )

    model, tokenizer = load_llm(config)
    max_new_tokens = config["generation"]["max_new_tokens"]

    result = generate_answer(
        question=question,
        retriever=retriever,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        use_router=use_router,
        index=index,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        top_chapters=top_chapters,
        route_pool=route_pool,
        retrieval_mode=mode,
        candidate_k=candidate_k,
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
