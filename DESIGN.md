# Design

This document records the main design decisions behind the system: the problem each one addresses, what was chosen, the alternatives considered, and the trade-offs accepted. The README describes what the system does and how to run it; this document explains why it is built this way.

## Goal

Answer questions about the EU AI Act (Regulation (EU) 2024/1689) from the text of the regulation itself. An answer is only useful if it can be checked, so the system has to:

- find the passages that contain the relevant rule, including its conditions and exceptions;
- cite the article behind every claim;
- say so when the retrieved material does not support an answer, instead of guessing.

The system also has to run as a service: built and tested automatically, deployed with one command, observable in production, and cheap to run on CPU.

## Decisions and hypotheses

Not every choice below has the same standing. Some follow directly from the structure of the problem; others are reasonable starting points that still need evidence. Keeping the two apart makes it clear what the evaluation has to prove.

| Category | Examples |
|---|---|
| Architectural decisions | Chunks respect article boundaries; retrieval and generation are separate stages; each model runs on its own inference server. |
| Implementation choices | `bge-base-en-v1.5`, BM25, Reciprocal Rank Fusion, routing parameters, chunk size. |
| Hypotheses to validate | Routing improves retrieval; hybrid search beats dense search; article-aware chunks beat fixed-size chunks; `bge-base` is a good fit compared with alternatives. |

Each hypothesis is tested against the golden set as a Langfuse dataset experiment, and the README reports the results.

## Retrieval and generation

### Chunks never cross an article boundary

**Problem.** The regulation is structured into chapters, articles, paragraphs and points, and that structure carries legal meaning. Splitting by size alone can separate an obligation from its condition, cut a list of requirements in half, separate an exception from the rule it applies to, or mix two articles in one chunk.

**Decision.** Split the text at article headings first, then split each article into chunks of up to 180 tokens with 10% overlap. Chapter headings are cut out so they do not end up in the previous article's last chunk. Every chunk carries its article, chapter and page numbers as metadata.

**Alternatives.** Fixed-size chunking over the whole text. It is still implemented (`strategy: sentence`), so the two can be compared on the same golden set.

**Trade-offs.** Articles vary widely in length, so long articles are still split by size and a single chunk may hold only part of an article. Short articles produce small chunks with little context.

### Embedding model: `BAAI/bge-base-en-v1.5`

**Problem.** Lexical search misses passages that use different words from the question. "Who is responsible for ensuring compliance?" should find "Providers shall ensure that their AI systems...".

**Decision.** Use `BAAI/bge-base-en-v1.5`, a model trained for retrieval. The base size is a balance between quality, memory and CPU cost. The system is scoped to English, the language of the source text, so an English model fits.

**Alternatives.** Larger BGE models, multilingual models, and hosted embedding APIs.

**Trade-offs.** This is a baseline choice, not a proven best; larger models are candidates to compare. Questions in other languages are out of scope and end in the escape phrase.

### Hybrid retrieval with Reciprocal Rank Fusion

**Problem.** Dense search finds paraphrases but can miss exact legal terms, article numbers and defined expressions. BM25 finds exact terms but misses paraphrases. In legal text, a passage can be relevant for either reason.

**Decision.** Run dense search and BM25, 15 candidates each, and combine the two rankings with Reciprocal Rank Fusion. The top 8 passages go to the prompt.

**Alternatives.** Dense search only (`mode: dense`, still available), or a weighted sum of scores.

**Why RRF.** It combines rankings by position, so scores on different scales never have to be normalized or weighted. There are no weights to tune on a small golden set.

**Trade-offs.** More work per query. RRF scores are small by construction and only express rank, so the score shown with each source should not be read as a relevance measure.

### Chapter router

**Problem.** Chapters cover distinct subjects. A search over the whole regulation can return passages that are semantically close to the question but come from an unrelated part of it.

**Decision.** Before retrieval, run a dense search over the whole index (30 candidates), score each chapter by its best candidate, and restrict the hybrid search to the top 2 chapters.

**Alternatives.** Global search with no routing (`use_router: false`), or a classifier or LLM that picks chapters.

**Trade-offs.** It adds a step and a failure mode: if the router picks the wrong chapters, the right passage never reaches retrieval. This is the hypothesis most in need of evidence, so retrieval is evaluated both with and without it.

### Fixed escape phrase

**Problem.** Retrieval can return passages that do not answer the question. A model that must always answer will fill the gap with outside knowledge or invented details.

**Decision.** The prompt defines one exact phrase, `Not covered in the provided material.`, for when the passages contain nothing relevant. When they cover only part of the question, the model answers that part and says what is missing. The API returns an `escaped` flag, and the flag is recorded in every trace.

**Trade-offs.** An escape means the context given to the model was not enough, not that the regulation does not cover the question. A high escape rate on in-scope questions points to retrieval or chunking, not to the model. Tracking the rate in Langfuse makes that visible.

### Prompt-injection safeguards

**Problem.** The question and the retrieved passages are both inserted into the prompt. Either can contain instructions: ignore the rules, reveal the prompt, answer from outside knowledge.

**Decision.** Several layers:

1. The instructions, the passages and the question are in separate, delimited sections.
2. Every inserted value is HTML-escaped, so it cannot close a delimiter.
3. The instructions say that passages are reference material, never instructions, and that commands inside them must be ignored.
4. Answers must be based only on the passages, with an article cited for every claim.
5. Questions are limited to 2,000 characters by the API.

**Trade-offs.** Prompting is defense in depth, not a guarantee: the model still reads the text semantically. The impact is bounded by design, because the model has no tools and no access to other systems, so a successful injection can only produce a wrong answer. Adversarial questions in the golden set are the next step.

### Baseline LLM

**Decision.** `Qwen2.5-0.5B-Instruct`, quantized to 4 bits on CPU, on purpose. A small model makes weaknesses in retrieval and prompting visible, keeps the CPU deployment cheap, and gives a fixed baseline to measure later improvements against. The model server is swappable, so a larger model on GPU is a configuration change.

## Serving

### Each model on its own inference server

**Problem.** Loading the embedding model and the LLM inside the API makes the image large (PyTorch alone is several gigabytes), slows startup, and ties the API's scaling to the models'.

**Decision.** Embeddings run on Text Embeddings Inference. The LLM runs on llama.cpp for CPU or vLLM for GPU, both behind the same OpenAI-compatible address. The API only orchestrates, and its image has no PyTorch.

**Alternatives.** Running the models in process with `transformers`, which is still possible for local experiments (`backend: hf`), or vLLM for both models.

**Trade-offs.** More containers to run and monitor. In exchange, each server is maintained upstream and optimized for its job, and the API image stays small.

### The same embedding server for ingestion and queries

**Decision.** The ingestion job and the API both embed through TEI with the same model. Documents and questions are embedded the same way, including the query instruction that BGE models expect, so the index and the queries can never drift apart.

### Separate liveness and readiness

**Decision.** `/health` only says the process is up. `/ready` says the vector collection exists and can be queried. If the collection is missing at startup, the API fails immediately with a message that says to run the ingestion job, instead of failing on the first question with an obscure error.

### Interface as a separate service

**Decision.** The Gradio UI is its own image and container, and calls the API over HTTP like any other client. The API stays free of UI dependencies, and the interface can be replaced without touching it.

**Alternatives.** Streamlit, or an HTML page served by the API.

## Delivery and operations

### Build once, deploy many

**Decision.** CI runs the tests, builds the images once, checks that the app imports inside the image and that the image stays under 1 GB, and only then publishes both images, tagged with the commit SHA and `latest`. The server never builds: it pulls a tested image. Deploying a commit and rolling back to an earlier one are the same command with a different tag.

### Registry: GHCR

**Decision.** Images are published to GitHub Container Registry with the workflow's own short-lived token. No credentials are stored in CI.

**Alternatives.** Amazon ECR. Publishing to ECR from GitHub Actions without long-lived keys needs OIDC federation between GitHub and AWS. Storing long-lived AWS access keys as CI secrets was ruled out, so ECR is planned together with OIDC.

### One deploy script, also used at boot

**Decision.** `scripts/deploy.sh` reads the secrets, pulls the images, starts the stack and waits for `/ready`. A systemd unit runs the same script when the instance starts, so starting the instance brings the service up, and the manual and automatic paths cannot diverge. An optional smoke test (`SMOKE=1`) asks one real question end to end.

### Secrets in SSM Parameter Store

**Decision.** The Langfuse keys are SecureString parameters, read at deploy time by the instance role. The role can read only this project's parameters.

**Alternatives.** An `.env` file on the server, which puts secrets on disk and has to be copied by hand.

### No exposed application ports

**Decision.** The API and the UI bind to `127.0.0.1` on the instance. Access is through a Session Manager port-forwarding session, authenticated with temporary AWS credentials, so no inbound rule is needed for the application.

### Configuration overlay

**Decision.** One base configuration for local use, plus a small Docker overlay that only changes addresses and backends, selected by an environment variable. The same image runs on a laptop and on AWS.

## Observability

Every request is traced in Langfuse, with spans for routing, retrieval and generation. The trace records the selected chapters, the retrieved passages, the full prompt, token usage and the `escaped` flag. Traces are tagged by environment (`production`, `development`, `evaluation`), so production traffic, manual tests and experiments stay separate in the same project.

## Scope of v1

- Questions and sources are in English.
- Answers come from the articles of the regulation. Indexing the annexes is planned.
- Retrieval is evaluated on a golden set of 23 questions with article labels. Out-of-scope and adversarial questions, MRR, and generation metrics are planned.
- The deployment is a single instance. A GPU benchmark, an evaluation gate and an end-to-end stack test in CI, ECR with OIDC, and Terraform are the next infrastructure steps.
