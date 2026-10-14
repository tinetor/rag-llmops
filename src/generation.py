from llm import load_llm
from prompts import ESCAPE_PHRASE, build_prompt
from retrieval import load_retriever, retrieve


def generate_answer(question: str, retriever, llm) -> dict:
    passages = retrieve(retriever, question)
    prompt = build_prompt(question, passages)
    answer = llm(prompt)
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
    llm = load_llm()

    question = "What are the prohibited AI practices?"
    result = generate_answer(question, retriever, llm)

    print(f"Q: {question}\n")
    print(f"A: {result['answer']}\n")
    print(f"escaped: {result['escaped']}")
    print("sources:")
    for s in result["sources"]:
        score = f"{s['score']:.3f}" if s["score"] is not None else "N/A"
        print(f"  page {s['page']} (score {score})")
