"""HTTP API for the RAG application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_llmops.rag_pipeline import (
    answer,
    close_components,
    is_ready,
    load_components,
)

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    text: str
    article: int | str | None = None
    chapter: int | str | None = None
    chapter_title: str | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    escaped: bool
    sources: list[Source]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models, index and clients once, before serving requests.
    app.state.components = load_components()
    yield
    # Shutdown: release clients and flush pending traces.
    close_components(app.state.components)


app = FastAPI(title="RAG LLMOps API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    """Liveness check: the process is up and the service finished loading."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness check: the vector collection is available to serve queries."""
    if not is_ready(app.state.components):
        raise HTTPException(status_code=503, detail="Vector index is not available.")
    return {"status": "ready"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        return answer(app.state.components, request.question)
    except Exception as exc:
        # Log the full error server-side; do not leak internals to the client.
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail="RAG query failed.") from exc
