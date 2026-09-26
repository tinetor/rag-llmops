# RAG Compliance Assistant — Design

## Motivation
Compliance team has 18 mandatory courses / 6 months, 3,000-page reference material.
Generic LLM fails: no access to internal/company-specific regulation.
Long-prompt approach ruled out: context window insufficient for 3,000 pages.
-> RAG chosen to fit the actual constraint, not as a default choice.

## Chunking
- Source: PDF only (no original .docx)
- Font-size heuristic detects headings (works because source is Word-authored)
- Heading-attached chunking: each chunk carries its nearest heading
- Small sections merged into the next, large sections split
- Target ~100-400 tokens per chunk

## Embeddings
- sentence-transformers family (fine-tuned for similarity search), not raw BERT/RoBERTa
- Baseline: all-MiniLM-L6-v2
- Config-driven, swappable without touching pipeline code

## Evaluation
- Golden set: 100 labeled Q&A pairs (of 1,000 total questions; other 900 are human study material, not used in eval)
- Two-stage funnel: (1) manual screening across model x chunk-size candidates, narrowed to ~2 finalists, (2) Recall@5 on golden set as primary metric, MRR as tiebreaker
- k=5 because that's exactly what's fed to the generator downstream
- Known risk: reusing the same 100 examples to both select config and report final number risks optimistic bias (selection effect) -> mitigated by narrowing configs aggressively in stage 1
- Golden set reused as fixed regression baseline on every prompt/model/chunk-size change

## Retrieval
- Dense-only (cosine similarity) for v1
- Improvement path: hybrid dense + BM25 via Reciprocal Rank Fusion (dense dilutes exact terms like policy names/section numbers)
- Reranking explicitly skipped at this scale (10 users, recall already acceptable)

## Generation & Grounding
- Full RAG: LLM generates conversational answer from top-k retrieved chunks
- Prompt constrains model to answer only from provided context
- Grounding instruction placed at both start AND end of prompt (mitigates "lost in the middle")
- Fixed escape phrase when unsupported -> doubles as a monitoring signal
- Source paragraph shown alongside generated answer for user verification

## Deployment
- FAISS index on a mounted volume, not baked into the app image
- Rebuild writes to a new file, atomic pointer swap on success only

## Production concerns
- API retries: exponential backoff, capped at 3 attempts
- Cost: no budget visibility from department -> track own usage internally regardless
- Monitoring: escape-phrase rate + groundedness judge flags as passive signals (critical given ~10 users -> sparse user feedback), traces for deep inspection

## Governance (open questions before build)
- Data can leave to external API: CONFIRMED
- Scope: currently ~10 users on this team
- Audit logging of queries: TBD
- Formal owner/approver: TBD
