"""Chat interface for the RAG API.

A separate app: it holds no models and no index, it only calls POST /query on
the API and shows the answer with the passages it was built from.
"""

import os

import gradio as gr
import httpx

API_URL = os.getenv("API_URL", "http://app:8080").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180"))
SOURCE_HEADERS = ["Art.", "Ch.", "Score", "Passage"]
PASSAGE_PREVIEW = 300


def source_rows(sources: list[dict]) -> list[list]:
    """Turn the API's sources into table rows, with shortened passages."""
    rows = []
    for source in sources:
        text = " ".join(source.get("text", "").split())
        if len(text) > PASSAGE_PREVIEW:
            text = text[:PASSAGE_PREVIEW].rstrip() + "..."
        score = source.get("score")
        rows.append([
            source.get("article") or "",
            source.get("chapter") or "",
            f"{score:.3f}" if isinstance(score, (int, float)) else "",
            text,
        ])
    return rows


def ask(question: str, history: list[dict]):
    """Send one question to the API and update the conversation and sources."""
    question = question.strip()
    if not question:
        return "", history, gr.update()

    history = history + [{"role": "user", "content": question}]
    try:
        response = httpx.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as exc:
        message = f"The API returned an error ({exc.response.status_code})."
        return "", history + [{"role": "assistant", "content": message}], []
    except httpx.HTTPError:
        message = "The API could not be reached. Is the stack running?"
        return "", history + [{"role": "assistant", "content": message}], []

    answer = body["answer"]
    if body.get("escaped"):
        answer += "\n\n(The retrieved material does not cover this question.)"
    history = history + [{"role": "assistant", "content": answer}]
    return "", history, source_rows(body.get("sources", []))


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Regulation Q&A") as demo:
        gr.Markdown(
            "# Regulation Q&A\n"
            "Answers are generated only from retrieved passages of the regulation, "
            "with the articles they come from. Each question is answered on its own. "
            "This is not legal advice."
        )
        with gr.Row():
            with gr.Column(scale=3):
                # Model output is shown as plain text, never rendered as HTML or Markdown.
                chat = gr.Chatbot(label="Conversation", height=480, render_markdown=False)
                question = gr.Textbox(
                    label="Question",
                    placeholder="e.g. Which AI practices are prohibited?",
                    max_length=2000,
                    lines=2,
                )
                with gr.Row():
                    send = gr.Button("Ask", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(scale=2):
                sources = gr.Dataframe(
                    headers=SOURCE_HEADERS,
                    label="Sources for the last answer",
                    interactive=False,
                    wrap=True,
                    column_widths=["11%", "11%", "15%", "63%"],
                )

        send.click(ask, [question, chat], [question, chat, sources])
        question.submit(ask, [question, chat], [question, chat, sources])
        clear.click(lambda: ("", [], []), None, [question, chat, sources])
    return demo


if __name__ == "__main__":
    build_ui().queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        footer_links=[],
    )
