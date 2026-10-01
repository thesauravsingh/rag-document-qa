# RAG Document Q&A

Production-style retrieval-augmented generation system: a curated AWS documentation corpus is chunked and embedded into pgvector, retrieved with hybrid search (semantic + BM25, fused with Reciprocal Rank Fusion), re-ranked, and answered with citations through a FastAPI service in Docker Compose. Includes an evaluation harness (Recall@k, MRR).

**Status:** planned — starts after the NYC Taxi Analytics Pipeline is finished.

## What's here

| File | Contents |
|---|---|
| `PROJECT_2_RAG_SYSTEM.md` | Build spec: architecture, phases, acceptance criteria |

## Setup

```bash
cp .env.example .env   # add your OPENAI_API_KEY
```
