# Enterprise RAG - Multi-Tenant HR & Document Copilot

A production-grade, multi-tenant RAG (Retrieval-Augmented Generation) system for querying enterprise documents and policies. Upload PDFs per tenant, ask questions in natural language, and receive precise, hallucination-resistant answers with source citations. Built with LangGraph agent routing and fully observable via Langfuse.

## Architecture

```
                                      ┌─────────────────────────────────────────┐
                                      │              LangGraph Agent            │
                                      │                                         │
Query ───────────────────────────────►│  [Retrieve Node]                        │
                                      │  ├─ pgvector HNSW (text-embedding-3-sm) │
                                      │  ├─ In-memory BM25 Keyword Search       │
                                      │  └─ Reciprocal Rank Fusion (RRF, k=60)  │
                                      │                    │                    │
                                      │                    ▼                    │
                                      │  [Classify Node] (gpt-4o-mini)          │
                                      │  Relevance check (filter off-topic)     │
                                      │          │                   │          │
                                      │     (On-topic)          (Off-topic)     │
                                      │          │                   │          │
                                      │          ▼                   ▼          │
                                      │   [Answer Node]       [Escalate Node]   │
                                      │   ├─ Cohere Rerank    └─ Polite HR      │
                                      │   │  (rerank-v3.0)       escalation     │
                                      │   └─ gpt-4o-mini         response       │
                                      │      + citations                        │
                                      └──────────┬───────────────────┬──────────┘
                                                 │                   │
                                                 ▼                   ▼
                                              Answer             Fallback
```

**Retrieval & Agent Pipeline:**
- **Vector search** — Cosine distance via pgvector HNSW index (`text-embedding-3-small`, 1536 dims, dynamic thresholding `min_similarity=0.25`).
- **BM25 keyword search** — Exact term matching via `rank-bm25` built over tenant chunks.
- **RRF (Reciprocal Rank Fusion)** — Merges dense semantic and sparse keyword rankings with $k=60$.
- **LangGraph Quality Classifier** — Validates candidate chunks via `gpt-4o-mini` to route queries: on-topic chunks proceed to answer generation; completely irrelevant queries trigger graceful escalation without hallucinations.
- **Cohere Reranker** — `rerank-english-v3.0` cross-encoder reranks top candidates for optimal context precision.
- **Answer Synthesis** — `gpt-4o-mini` with strict ground-truth constraints and per-chunk document/page attribution.

**Stack:** FastAPI · PostgreSQL 16 + pgvector · LangGraph · Langfuse · Cohere · OpenAI · SQLAlchemy (async) · Alembic · Docker Compose

---

## Eval Results

Evaluation benchmark across enterprise documents:

| Metric | Score |
|---|---|
| Retrieval accuracy | 100% |
| Answer accuracy | 90% |
| Test dataset | 100 documents |

Run the eval suite yourself:
```bash
python scripts/run_eval.py --tenant-id <uuid>
```

---

## Observability (Langfuse)

All queries are traced end-to-end using [Langfuse](https://cloud.langfuse.com):
- **Traces & Spans** — Tracks query flow through retrieval, classification, reranking, and generation.
- **Latency & Cost** — Monitors token consumption, response latency, and per-tenant costs.
- **Debug & Inspect** — View retrieved chunks, prompt payloads, and agent routing decisions in real time.

---

## Setup

### Prerequisites

- Docker + Docker Compose
- OpenAI API key
- Cohere API key
- Langfuse API keys (public & secret key from [cloud.langfuse.com](https://cloud.langfuse.com))

### 1. Configure environment

Create a `.env` file in the project root:

```env
POSTGRES_USER=hrcopilot
POSTGRES_PASSWORD=...
POSTGRES_DB=hrcopilot
DATABASE_URL=postgresql+asyncpg://{user}:{password}@db:5432/{db}

OPENAI_API_KEY=sk-...
COHERE_API_KEY=...

# Langfuse Observability
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

PGADMIN_DEFAULT_EMAIL=...
PGADMIN_DEFAULT_PASSWORD=...
```

> `.env` is gitignored — never commit it.
> The `POSTGRES_PASSWORD` and pgAdmin credentials (`admin@admin.com` / `admin` by default) are for local development only.

### 2. Start services

```bash
docker compose up --build
```

This starts:
- `db` — PostgreSQL 16 with pgvector on port `5432` (or external `5433` if mapped)
- `api` — FastAPI on port `8000` with hot reload
- `pgadmin` — pgAdmin 4 on port `5050`

### 3. Run migrations

```bash
docker compose exec api alembic upgrade head
```

### 4. Verify

- Swagger API docs: `http://localhost:8000/docs`
- Langfuse Traces: `https://cloud.langfuse.com`
- pgAdmin: `http://localhost:5050`

---

## API

### Create a tenant

```bash
curl -X POST http://localhost:8000/api/v1/tenants/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp"}'
```

### Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "tenant_id=<uuid>" \
  -F "file=@/path/to/document.pdf"
```

### Query

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "<uuid>",
    "question": "What is the notice period for termination?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "The agreement requires 14 days notice for termination.",
  "sources": [
    {
      "document_title": "Contract.pdf",
      "page_number": 3,
      "content": "...",
      "rrf_score": 0.032787,
      "relevance_score": 0.891234
    }
  ]
}
```

`rrf_score` — combined rank from vector + BM25 fusion (higher = retrieved by more signals).  
`relevance_score` — Cohere cross-encoder score (semantic relevance to the question).

---

## Database

pgAdmin is available at `http://localhost:5050`:

| Field | Value |
|---|---|
| Login email | `admin@admin.com` |
| Login password | set in `docker-compose.yml` |
| Host | `db` |
| Port | `5432` |
| Database | set in `.env` (`POSTGRES_DB`) |
| Username | set in `.env` (`POSTGRES_USER`) |
| Password | set in `.env` (`POSTGRES_PASSWORD`) |

---

## Evaluation

The repo includes test runners and benchmark datasets to validate retrieval accuracy and answer quality:

### 1. Live API Evaluation (`scripts/run_eval.py`)
Queries the running FastAPI service, verifies retrieved document titles/pages, judges answer accuracy using `gpt-4o-mini` as an LLM judge, and generates timestamped summary reports.

```bash
# Run against the 3-document target benchmark (100% pass suite)
python scripts/run_eval.py --eval-file scripts/run_eval_3.json --tenant-id <uuid>

# Run against the full legal QA dataset (with document auto-filtering)
python scripts/run_eval.py --eval-file scripts/legal_eval.json --tenant-id <uuid> --limit 15
```

Reports are automatically saved to `eval_results/`.

### 2. Direct Pipeline Evaluation (`eval_runner.py`)
Directly runs the query service against the database session without requiring network calls to FastAPI:

```bash
# Inside docker container
docker compose exec api python eval_runner.py --tenant-id <uuid>

# Or locally
python eval_runner.py --tenant-id <uuid> --eval-file eval.json
```

**Scoring Methodology:**
- **Retrieval scoring** — Verifies whether the expected document stem/title appears in the top retrieved sources.
- **Answer scoring** — LLM-as-a-judge (`gpt-4o-mini`) inspects the question, expected answer, retrieved snippets, and actual answer to render a `PASS`/`FAIL` verdict based on semantic truth.
- **Negative tests** (`should_retrieve: false`) — Confirms that out-of-scope questions route to the escalation path rather than hallucinating facts.

---

## Project Structure

```
hr-copilot/
├── app/
│   ├── agent/                  # LangGraph agent implementation
│   │   ├── graph.py            # StateGraph definition & routing
│   │   ├── nodes.py            # retrieve, classify, answer, escalate nodes
│   │   └── state.py            # AgentState schema
│   ├── api/v1/endpoints/       # tenants, documents, query
│   ├── core/config.py          # Pydantic settings & env loading
│   ├── db/session.py           # Async SQLAlchemy session engine
│   ├── models/                 # Tenant, Document, Chunk (pgvector)
│   ├── schemas/                # Pydantic request/response models
│   └── services/
│       ├── parsing.py          # PDF → pages → chunks
│       ├── embedding_service.py # OpenAI embeddings (Langfuse instrumented)
│       ├── bm25_service.py     # BM25 in-memory keyword search
│       ├── rrf.py              # Reciprocal Rank Fusion
│       ├── reranker_service.py # Cohere cross-encoder reranker
│       └── query_service.py    # Direct query orchestration
├── alembic/                    # DB migrations
├── eval.json                   # HR baseline eval dataset (10 questions)
├── eval_results/               # Eval benchmark reports
├── eval_runner.py              # Direct DB eval runner
├── scripts/
│   ├── run_eval.py             # Live API eval runner with auto-filtering
│   ├── run_eval_3.json         # 3-document target legal benchmark (9 questions)
│   ├── legal_eval.json         # Full legal Q&A dataset (50 documents)
│   ├── download_50_pdfs.py     # PDF dataset scraper
│   └── ingest_test_pdfs.py     # Batch ingestion script
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
