# Build Spec: RAG-Powered Document Q&A System

> A production-style Retrieval-Augmented Generation pipeline demonstrating end-to-end AI/LLM engineering skills.

## Project Goal (One Sentence)

Build a containerized RAG system that ingests a real document corpus, chunks and embeds it into pgvector, retrieves relevant context via hybrid (semantic + keyword) search with re-ranking, and answers user questions via an OpenAI LLM exposed through a FastAPI endpoint — with evaluation metrics to prove it works.

## Why This Project Exists

"AI Data Engineer" roles in 2026 expect candidates to understand the data engineering side of LLM applications: chunking strategies, vector storage, retrieval quality, evaluation, and production concerns. This project demonstrates all of those without requiring novel ML research — it's the *infrastructure* of an LLM app, which is exactly what data engineers own.

## What Done Looks Like

- [ ] GitHub repo with clean structure, README, and architecture diagram
- [ ] Working ingestion pipeline that processes a real corpus (50-200 documents)
- [ ] pgvector running in Docker Compose
- [ ] Hybrid retrieval implementation (semantic + BM25 keyword)
- [ ] Re-ranking step (LLM-based or cross-encoder)
- [ ] FastAPI endpoint with proper request/response models
- [ ] Docker Compose orchestrating Postgres + API service
- [ ] Evaluation set of 20+ Q&A pairs with retrieval accuracy metrics
- [ ] README with architecture diagram, sample queries, eval results
- [ ] Demo gif or 5-10 sample API responses captured in docs

## Tech Stack & Why

| Tool | Why this choice |
|---|---|
| **Python 3.11** | Industry standard for LLM work |
| **OpenAI API** | `text-embedding-3-small` is cheap and good; gpt-4o-mini for generation is cheap and good |
| **pgvector** | Postgres extension — more impressive than Pinecone because it shows you understand the underlying mechanics. Also free. |
| **LangChain** | Industry standard for RAG orchestration; not glamorous but expected |
| **rank_bm25** | Pure Python BM25 implementation; pairs well with semantic for hybrid |
| **FastAPI** | Async, type-safe, auto-OpenAPI docs |
| **Docker Compose** | Reproducible env, demonstrates containerization skill |
| **pytest** | For the eval suite |

## Corpus: AWS Service Documentation (Curated Subset)

**Why this corpus:**
- Public, no auth needed
- Niche enough that vanilla ChatGPT gives generic/outdated answers
- Real technical content (not trivia) where RAG genuinely helps
- Easy to construct good Q&A eval pairs
- Recruiters and interviewers will recognize the content

**Alternative corpora (pick one):**
1. **AWS docs** — Recommended. Use the docs for 3-4 specific services (e.g., Glue, Redshift, MWAA, Athena — core data-engineering services)
2. **dbt docs** — All of docs.getdbt.com (good if you want analytics engineering signal)
3. **arxiv abstracts** — Last 6 months of cs.LG papers (good if you want ML research signal)
4. **SEC 10-K filings** — 5 specific companies (good if you want financial/business signal)

**Recommendation: Use AWS docs for 4 data-engineering services — Glue, Redshift, MWAA, Athena.** These are widely used in data platforms, so the corpus and the questions about it are realistic.

**How to get the corpus:**
- Scrape ~50-200 pages from `docs.aws.amazon.com/glue/`, `docs.aws.amazon.com/redshift/`, etc.
- Use a polite scraper (rate-limited, user-agent set)
- Or: download AWS docs as Markdown from their open-source docs GitHub repo (cleaner)
- Target ~50-200 documents, ~500K-2M tokens total

## Project Structure

```
rag-aws-docs/
├── README.md
├── ARCHITECTURE.md
├── .env.example                       # Template for OPENAI_API_KEY, etc.
├── .gitignore                         # Never commit .env, raw docs, embeddings
├── requirements.txt
├── docker-compose.yml                 # pgvector + API service
├── Dockerfile                         # For API service
│
├── ingestion/
│   ├── __init__.py
│   ├── scrape_docs.py                 # Pulls AWS docs to local corpus/
│   ├── chunk.py                       # Text chunking logic
│   ├── embed.py                       # Calls OpenAI embeddings API
│   └── load_to_pgvector.py            # Loads embeddings into Postgres
│
├── retrieval/
│   ├── __init__.py
│   ├── semantic.py                    # pgvector cosine similarity search
│   ├── keyword.py                     # BM25 search
│   ├── hybrid.py                      # Combines + reciprocal rank fusion
│   └── rerank.py                      # LLM-based re-ranking
│
├── generation/
│   ├── __init__.py
│   ├── prompts.py                     # System prompts, RAG prompt template
│   └── chain.py                       # LangChain RAG chain
│
├── api/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app
│   ├── models.py                      # Pydantic request/response
│   └── routes.py
│
├── eval/
│   ├── eval_set.json                  # 20+ Q&A pairs with expected sources
│   ├── run_eval.py                    # Runs eval and produces metrics
│   └── results/
│       └── eval_results.md            # Latest eval results
│
├── notebooks/
│   ├── 01_explore_corpus.ipynb        # Initial corpus analysis
│   ├── 02_chunking_experiments.ipynb  # Find best chunk size
│   └── 03_retrieval_quality.ipynb     # Compare semantic vs hybrid
│
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── corpus/                            # Raw scraped docs (gitignored)
│   └── (not committed to git)
│
└── docs/
    ├── architecture.png               # Excalidraw diagram
    └── demo.gif                       # API in action
```

## Build Order (Do These In Order)

### Phase 1: Foundation (Day 1, ~3 hours)

**1.1 Project setup**
- Create GitHub repo `rag-aws-docs`
- `python -m venv venv && source venv/bin/activate`
- `pip install langchain langchain-openai langchain-postgres pgvector psycopg2-binary fastapi uvicorn rank_bm25 pytest python-dotenv beautifulsoup4 requests`
- Save `requirements.txt`
- Create `.env.example` with `OPENAI_API_KEY=`
- Create `.gitignore` for Python + `.env` + `corpus/` + `*.pkl`

**1.2 OpenAI setup**
- Get API key from platform.openai.com
- Add $10 credit (you'll spend ~$2-5 for this project total)
- Set spending limit to $25 to be safe
- Save key to `.env` locally (NEVER commit)

**1.3 pgvector via Docker**
- Create `docker-compose.yml`:
  ```yaml
  services:
    postgres:
      image: pgvector/pgvector:pg16
      environment:
        POSTGRES_PASSWORD: ragpass
        POSTGRES_DB: ragdb
      ports: ["5432:5432"]
      volumes:
        - pgdata:/var/lib/postgresql/data
  volumes:
    pgdata:
  ```
- `docker compose up -d`
- Connect with `psql` or DBeaver, run `CREATE EXTENSION vector;`
- Create chunks table:
  ```sql
  CREATE TABLE doc_chunks (
    id SERIAL PRIMARY KEY,
    source_url TEXT NOT NULL,
    source_doc TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
  );
  CREATE INDEX ON doc_chunks USING hnsw (embedding vector_cosine_ops);
  ```

**Acceptance criteria for Phase 1:**
- [ ] Repo exists, deps installed
- [ ] OpenAI API key works (test with 1 embedding call)
- [ ] pgvector container running, vector extension enabled
- [ ] Connection from Python works

### Phase 2: Corpus Acquisition (Day 1-2, ~2 hours)

**2.1 Choose your 4 AWS services** (Glue, Redshift, MWAA, Athena recommended)

**2.2 Scrape or download**
- **Option A (easier):** Clone AWS docs GitHub repo, copy relevant Markdown files
  - Check `github.com/awsdocs` for source repos
- **Option B:** Polite scraper
  - `scrape_docs.py` uses `requests` + `BeautifulSoup`
  - 1-second delay between requests
  - Set User-Agent
  - Save raw HTML/Markdown to `corpus/{service}/`
- Target 30-50 docs per service, 120-200 total

**2.3 Clean**
- Strip HTML tags if needed
- Remove nav/sidebar/footer content
- Keep only main content + headings
- Preserve source URL as metadata

**Acceptance criteria for Phase 2:**
- [ ] `corpus/` has 120-200 cleaned text/markdown files
- [ ] Each file has source URL traceable
- [ ] Total corpus is ~500K-2M tokens

### Phase 3: Chunking + Embedding (Day 2-3, ~3 hours)

**3.1 Chunking strategy**
- Use LangChain's `RecursiveCharacterTextSplitter`
- Initial config: chunk_size=512 tokens, overlap=50
- Use `tiktoken` to count tokens accurately
- Preserve document metadata (source, title, section)
- **Don't split mid-code-block** — add custom separator handling

**3.2 Embedding**
- Use `text-embedding-3-small` (1536 dim, $0.02/1M tokens — very cheap)
- Batch requests (100 chunks per API call)
- Add retry with exponential backoff
- Save to pgvector as you go (don't hold everything in memory)
- Log: chunks created, tokens embedded, $ spent estimate

**3.3 Experiment in notebook**
- `notebooks/02_chunking_experiments.ipynb`
- Try chunk sizes 256, 512, 1024
- For each, embed 50 sample chunks and test retrieval on 5 known queries
- Pick winner, document reasoning in README

**Acceptance criteria for Phase 3:**
- [ ] pgvector has 2K-10K embedded chunks
- [ ] Each chunk has source URL + service tag + chunk index
- [ ] Notebook documents chunk size choice
- [ ] Total embedding cost logged (<$1)

### Phase 4: Semantic Retrieval (Day 3, ~2 hours)

**4.1 Basic semantic search** — `retrieval/semantic.py`
- Function: `semantic_search(query: str, k: int = 10) -> List[Chunk]`
- Embed query → cosine similarity search in pgvector
- Return top-k chunks with scores
- Add filter capability (e.g., `service='glue'`)

**4.2 Test manually**
- Run 5-10 sample queries
- Inspect results — are they relevant?
- If results are bad, debug: chunking too small? metadata wrong? wrong embedding model?

**Acceptance criteria for Phase 4:**
- [ ] Semantic search returns relevant chunks for sample queries
- [ ] Latency <200ms per query (pgvector with HNSW should be fast)

### Phase 5: Hybrid Retrieval (Day 4, ~3 hours)

This is the **differentiator** — most tutorials skip this. Do it.

**5.1 BM25 keyword index** — `retrieval/keyword.py`
- Build BM25 index over all chunks using `rank_bm25`
- Persist index to disk (pickle) so you don't rebuild every startup
- Function: `keyword_search(query: str, k: int = 10) -> List[Chunk]`

**5.2 Hybrid combination** — `retrieval/hybrid.py`
- Implement **Reciprocal Rank Fusion (RRF)**:
  ```
  score(chunk) = sum(1 / (k + rank_in_each_method))
  ```
  Standard `k=60` works well
- Returns top-k merged results

**5.3 Why this matters (for README + interviews):**
- Semantic search misses exact-match queries ("How do I set ENABLE_VECTORIZATION?")
- Keyword search misses paraphrases ("How do I make queries faster?")
- Hybrid catches both. Real-world RAG systems use this.

**Acceptance criteria for Phase 5:**
- [ ] Both BM25 and semantic working independently
- [ ] Hybrid implementation tested
- [ ] Notebook shows hybrid beating semantic-only on 5+ queries

### Phase 6: Re-ranking (Day 4-5, ~2 hours)

**6.1 LLM-based re-ranker** — `retrieval/rerank.py`
- Take top-20 from hybrid → re-rank with LLM
- Prompt: "Rate the relevance of each passage to the query on a 1-10 scale"
- Use `gpt-4o-mini` for cost ($0.15/1M input tokens)
- Return top-5 after re-ranking

**6.2 Alternative: Cross-encoder**
- If you want to avoid extra LLM calls, use `sentence-transformers/ms-marco-MiniLM-L-6-v2`
- Cheaper and offline-capable, but worse quality
- Mention in README that you considered both

**Acceptance criteria for Phase 6:**
- [ ] Re-ranker improves quality on hard queries (document in eval)

### Phase 7: Generation + LangChain Chain (Day 5, ~2 hours)

**7.1 Prompt engineering** — `generation/prompts.py`
- System prompt: explicit "answer only from context, cite sources, say IDK if no answer"
- User prompt template includes: question + numbered context chunks
- Output format: answer + cited source URLs

**7.2 RAG chain** — `generation/chain.py`
- LangChain `RunnableParallel` or just plain functions (LangChain is overhead-y for simple chains)
- Steps: retrieve → rerank → format prompt → LLM call → parse response

**7.3 Test**
- Manual: ask 10 questions, check answers are grounded and cited

**Acceptance criteria for Phase 7:**
- [ ] End-to-end question → answer with citations works
- [ ] Answers include source URLs

### Phase 8: FastAPI Service (Day 5-6, ~3 hours)

**8.1 API design** — `api/main.py`
```
POST /query
  body: { "question": "...", "filters": {"service": "glue"}, "k": 5 }
  returns: { "answer": "...", "sources": [...], "retrieval_metadata": {...} }

GET /health
  returns: { "status": "ok", "pgvector_chunks": 1234 }

GET /metrics (optional)
  returns: latency stats, cost estimate
```

**8.2 Implementation**
- Pydantic models for request/response
- Async endpoint (FastAPI default)
- Error handling for OpenAI failures, pgvector failures
- Logging (structlog or just stdlib)
- Add basic observability: log query, retrieval count, latency, tokens used

**8.3 Containerize**
- `Dockerfile` for the API
- Add API service to `docker-compose.yml`
- `docker compose up` should bring up Postgres + API
- Test: `curl localhost:8000/query -d '...'`

**Acceptance criteria for Phase 8:**
- [ ] `docker compose up` brings up full stack
- [ ] API responds to queries with cited answers
- [ ] Auto-generated OpenAPI docs at `/docs`

### Phase 9: Evaluation (Day 6-7, ~3 hours)

This is where you separate yourself from tutorial-level work.

**9.1 Build eval set** — `eval/eval_set.json`
- 20+ Q&A pairs with expected source URLs
- Mix of:
  - Easy (single-doc answer)
  - Medium (multi-doc synthesis)
  - Hard (paraphrase/edge case)
  - Adversarial (questions where the answer ISN'T in the corpus — should return "I don't know")

Example entry:
```json
{
  "question": "What is the default Spark version in AWS Glue 4.0?",
  "expected_source": "docs.aws.amazon.com/glue/latest/dg/release-notes.html",
  "expected_answer_contains": ["Spark 3.3"],
  "difficulty": "easy",
  "type": "factual"
}
```

**9.2 Eval runner** — `eval/run_eval.py`
- For each Q in eval set:
  - Retrieve via your hybrid+rerank pipeline
  - Check: is expected_source in retrieved top-k? (Recall@k)
  - Generate answer
  - Check: does answer contain expected substrings? (rough quality proxy)
- Output markdown report with metrics

**9.3 Metrics to report**
- **Recall@5**: % of queries where expected source was retrieved in top-5
- **Recall@10**: same for top-10
- **MRR** (Mean Reciprocal Rank): inverse of position of correct doc
- **Answer quality**: manual 1-5 scoring on 20% of queries
- **Cost per query**: tokens × prices

**9.4 Iterate**
- Compare: semantic-only vs hybrid vs hybrid+rerank
- Document the comparison in README — this is gold for interviews

**Acceptance criteria for Phase 9:**
- [ ] eval_set.json has 20+ diverse Q&A pairs
- [ ] Eval runner produces metrics report
- [ ] Recall@5 >70% on the easy/medium set
- [ ] At least one adversarial query correctly returns "I don't know"

### Phase 10: Documentation & Polish (Day 7-8, ~3 hours)

**10.1 Architecture diagram**
- Excalidraw: Corpus → Chunker → Embedder → pgvector / BM25 → Hybrid Retriever → Re-ranker → LLM → Response
- Save to `docs/architecture.png`

**10.2 Demo gif**
- Use `asciinema` for terminal recording, or screen-record API calls
- ~30 seconds showing a question → answer with sources

**10.3 README**
- See template below

**10.4 Optional: deploy to free tier**
- Render.com or Fly.io has free tiers; deploy the API
- Live demo URL is gold

## README Template — Use This Structure

```markdown
# RAG-Powered AWS Documentation Q&A

A production-style Retrieval-Augmented Generation system that answers questions about AWS Glue, Redshift, MWAA, and Athena using semantic + keyword hybrid retrieval, LLM re-ranking, and citation-grounded generation.

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688)](https://fastapi.tiangolo.com/)
[![pgvector](https://img.shields.io/badge/pgvector-0.6+-336791)](https://github.com/pgvector/pgvector)

## 🎬 Demo
![Demo](docs/demo.gif)

[Live API →](your-deployment-url) (if deployed)

## 🏗️ Architecture
![Architecture](docs/architecture.png)

## 🎯 What This Demonstrates

- **Data ingestion at scale**: 200+ documents, 5K+ chunks, embedded into pgvector
- **Hybrid retrieval**: Semantic (pgvector HNSW) + keyword (BM25) merged via Reciprocal Rank Fusion
- **LLM re-ranking**: Improves Recall@5 from 71% → 84% on eval set
- **Production patterns**: FastAPI, Docker Compose, structured logging, evaluation harness
- **Honest evaluation**: 20+ Q&A pairs including adversarial queries

## 📊 Evaluation Results

| Configuration | Recall@5 | MRR | Avg Cost / Query |
|---|---|---|---|
| Semantic only | 71% | 0.58 | $0.0008 |
| Hybrid (BM25 + semantic) | 78% | 0.66 | $0.0008 |
| Hybrid + LLM re-rank | **84%** | **0.74** | $0.003 |

*(Your real numbers go here)*

Full eval methodology and per-query breakdown in `eval/results/`.

## 🛠️ Tech Stack
- **Python 3.11**
- **OpenAI** — `text-embedding-3-small` for embeddings, `gpt-4o-mini` for generation/re-ranking
- **pgvector** — HNSW-indexed vector search in Postgres
- **rank_bm25** — Keyword retrieval
- **LangChain** — RAG orchestration
- **FastAPI** + **Pydantic** — Async API layer
- **Docker Compose** — Orchestration

## 🚀 Quickstart

```bash
git clone https://github.com/yourusername/rag-aws-docs
cd rag-aws-docs
cp .env.example .env  # add your OPENAI_API_KEY
docker compose up -d
python ingestion/load_to_pgvector.py  # one-time corpus load
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I enable encryption on a Glue catalog?"}'
```

## 📁 Repository Structure
[paste tree]

## 💡 Key Design Decisions

**Why hybrid retrieval over semantic-only?**
Semantic search struggles with exact-match technical queries (API names, configuration flags). BM25 catches those. Reciprocal Rank Fusion is mathematically clean and parameter-free.

**Why pgvector over Pinecone/Weaviate?**
pgvector is open-source, runs locally, and Postgres familiarity transfers everywhere. Pinecone/Weaviate are managed but lock you into vendor APIs.

**Why chunk size 512?**
Tested 256, 512, 1024 (see `notebooks/02_chunking_experiments.ipynb`). 512 was best balance of context richness vs retrieval precision for technical docs.

**Why gpt-4o-mini and not gpt-4o?**
Cost ratio is ~20x. On AWS docs, mini is sufficient — no nuanced reasoning required, just synthesis of retrieved facts.

## 📚 What I Learned
[Fill in honestly after building]

## 🔮 Future Improvements
- [ ] Streaming responses (SSE) for better UX
- [ ] Query rewriting (HyDE) for ambiguous questions
- [ ] Caching layer (Redis) for common queries
- [ ] Multi-modal: index AWS docs screenshots/diagrams too
```

## Common Pitfalls (Avoid These)

1. **Committing your `.env` or API keys** — Triple-check `.gitignore` BEFORE first push
2. **Embedding the entire corpus on every code change** — Cache locally, version embeddings
3. **Ignoring chunking quality** — A bad chunker (splits mid-code-block, drops headers) breaks everything downstream. Spend time here.
4. **Skipping evaluation** — "I'll add evals later" → you won't. Build eval set in Phase 2, run it after each retrieval change.
5. **Trying to handcraft a "perfect" prompt** — Iterate on the system prompt only after retrieval is solid. Bad retrieval can't be saved by good prompts.
6. **Over-relying on LangChain abstractions** — LangChain is a research library disguised as a framework. Use it for chunking + the basic chain, but write retrieval/embedding yourself so you understand it.
7. **Not handling "no answer" cases** — RAG systems hallucinate when they should say "I don't know." Add explicit instruction in system prompt + at least 3 adversarial eval queries.

## Interview Defense Notes (Be Ready For These)

**Q: "Walk me through the architecture."**
*A: 60-second pitch: corpus → chunk → embed → pgvector + BM25 → hybrid retrieve → LLM re-rank → generate with citations. Mention scale (200 docs, 5K chunks), key insight (hybrid beats semantic-only by 7pts), and the eval set as proof.*

**Q: "Why pgvector vs Pinecone?"**
*A: Open-source, local-first, no vendor lock-in. Postgres skills transfer everywhere. For production at scale, Pinecone's managed indexing might win, but for this project pgvector with HNSW is plenty fast (<50ms p95).*

**Q: "How did you decide on chunk size?"**
*A: I ran experiments (show notebook). 256 was too small — chunks lacked context. 1024 was too big — retrieval became coarse. 512 hit the sweet spot for AWS docs which have section-level structure.*

**Q: "What's your eval methodology?"**
*A: 20+ hand-crafted Q&A pairs across difficulty levels including adversarial ("not in corpus") queries. Primary metrics are Recall@k and MRR for retrieval, plus manual answer quality scoring on a subset. I track this per-config to measure each component's contribution.*

**Q: "How would you productionize this?"**
*A: (a) Async background re-embedding pipeline as corpus updates, (b) Redis cache for common queries, (c) move to managed Postgres or Pinecone at scale, (d) add structured observability (latency, retrieval quality drift, cost per query), (e) add per-user rate limiting, (f) gating on answer quality (refuse low-confidence retrievals), (g) full CI/CD with eval suite as a quality gate.*

**Q: "What went wrong while building this?"**
*A: Have a real answer. Maybe: "My first chunker split AWS docs mid-code-block, which crushed retrieval quality on configuration questions. Had to add custom separator handling for fenced code blocks. Recall@5 jumped from 52% to 71% just from that fix."*

**Q: "Why this corpus and not your company's docs?"**
*A: Public so I could share the project; specifically AWS Glue/Redshift/MWAA/Athena because those are tools I use daily — I could write better eval questions and judge answer quality.*

## Budget Estimate

- OpenAI embeddings: ~$0.50 (1M tokens × $0.02/M × 25 = $0.50)
- OpenAI generation during dev + eval: ~$2-5
- pgvector (local Docker): $0
- GitHub: $0
- Optional deployment (Render/Fly free tier): $0
- **Total: ~$3-6**

## Timeline Reality Check

- **Aggressive:** 5-7 days
- **Realistic (10-12 hours/week):** 2-3 weeks
- **Padded:** 3-4 weeks

Aim for 2 weeks AFTER finishing Project 1.

---

**Next:** Build this with Claude Code. Hand it this README and ask it to start at Phase 1. Iterate phase-by-phase. Common failure mode: trying to skip evaluation. Don't.
