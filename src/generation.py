from langfuse import get_client

from llm import load_llm, generate_text, get_model_name
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
    top_k: int = 5,
    top_chapters: int = 2,
    route_pool: int = 30,
    retrieval_mode: str = "dense",
    candidate_k: int = 20,
) -> dict:
    langfuse = get_client()
    model_name = get_model_name(model)

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

            # Build retriever using the configured top_k
            retriever_kwargs = {
                "index": index,
                "persist_dir": persist_dir,
                "collection_name": collection_name,
                "k": top_k,
                "mode": retrieval_mode,
                "candidate_k": candidate_k,
            }

            if chapters:
                retriever_kwargs["chapter"] = chapters

            question_retriever = build_retriever(
                **retriever_kwargs
            )

        else:
            question_retriever = retriever

        # 2. Retrieve relevant chunks
        with langfuse.start_as_current_observation(
            as_type="span",
            name="retrieval",
            input={
                "question": question,
                "use_router": use_router,
                "top_k": top_k,
                "candidate_k": candidate_k,
            },
        ) as retrieval_span:

            passages = retrieve(
                question_retriever,
                question,
            )

            retrieval_span.update(
                output={
                    "chunk_count": len(passages),
                    "chunks": passages,
                }
            )

        # 3. Build the prompt with the retrieved context
        prompt = build_prompt(question, passages)

        # 4. Generate the answer with Hugging Face
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="generate-answer",
            model=model_name,
            input={"prompt": prompt},
            model_parameters={
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            },
        ) as generation:

            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )

            answer = generated["answer"]

            generation.update(
                input={"prompt": generated["chat"]},
                output=answer,
                usage_details={
                    "input": generated["input_tokens"],
                    "output": generated["output_tokens"],
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
