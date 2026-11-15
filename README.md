# Enterprise RAG - Multi-Tenant HR & Document Copilot

A multi-tenant RAG (Retrieval-Augmented Generation) system for querying enterprise documents and policies. Upload PDFs per tenant, ask questions in natural language, and receive grounded answers with document and page citations. Built on hybrid retrieval, LangGraph agent routing and an evaluation harness that measures retrieval and answer quality on every change.

## Architecture

```
                                      ┌─────────────────────────────────────────┐
                                      │              LangGraph Agent            │
                                      │                                         │
Query ───────────────────────────────►│  [Retrieve Node]                        │
                                      │  ├─ pgvector HNSW (text-embedding-3-sm) │
                                      │  ├─ Postgres full-text search (OR)      │
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

**Ingestion pipeline:**
- **Section-aware chunking** — PDFs are split along their numbered sections (`15. Budget, Finance and Audit.-`), across page breaks. Headings are accepted only as part of the longest run that numbers in order, which tolerates OCR-damaged headings and ignores sections quoted inside amendments. Short sections are merged, long ones are split at subsection boundaries, and documents without a usable section structure fall back to fixed-size chunks.
- **Document titles** — The document's real title is extracted from its opening text with `gpt-4o-mini`, since uploaded filenames are often meaningless.
- **Chunk headers** — Every chunk is prefixed with its document and section, e.g. `[The Virtual Assets Act, 2026 — s.15 Budget, Finance and Audit]`, so embeddings, keyword search and the answer prompt all know where a chunk belongs.

**Retrieval & agent pipeline:**
- **Vector search** — Cosine distance via pgvector HNSW index (`text-embedding-3-small`, 1536 dims, `min_similarity=0.25`).
- **Keyword search** — Postgres full-text search over a generated `tsvector` column (GIN index). Query terms are OR-ed and ranked with length-normalised `ts_rank`, so a relevant chunk is not dropped for missing a single query word.
- **RRF (Reciprocal Rank Fusion)** — Merges the semantic and keyword rankings with $k=60$.
- **Relevance classifier** — `gpt-4o-mini` checks the candidate chunks: on-topic queries proceed to answer generation; off-topic queries are escalated instead of answered.
- **Cohere reranker** — `rerank-english-v3.0` reorders the candidates. Rate-limit responses are retried; any fallback to the unranked order is logged (`RERANK_FALLBACK`).
- **Answer synthesis** — `gpt-4o-mini`, constrained to the retrieved context, with per-chunk document and page attribution.

**Stack:** FastAPI · PostgreSQL 16 + pgvector · LangGraph · Langfuse · Cohere · OpenAI · SQLAlchemy (async) · Alembic · Docker Compose

---

## Eval Results

Benchmark: 160 questions (144 answerable, 16 unanswerable) over 50 Pakistani federal legal documents — gazette PDFs from 2 to 839 pages, many of them OCR scans. Questions are sampled across each document, name the document they refer to, and carry an evidence quote; see [Evaluation](#evaluation).

| Metric | Score |
|---|---|
| Answer accuracy | 152/160 (95%) |
| Correct document in top 5 sources | 142/144 (99%) |
| Evidence passage in top 5 sources | 130/144 (90%) |
| Unanswerable questions answered anyway | 1/16 |

Contribution of each change, measured on the same question set:

| Configuration | Answer accuracy |
|---|---|
| Baseline (fixed-size chunks, AND keyword search) | 81% |
| + OR keyword search, document-title chunk headers | 88% |
| + Section-aware chunking | 92.5% |
| + Reranking applied to every query | 95% |

The first two rows were graded with `gpt-4o-mini` before the eval set was corrected; the remaining rows use the corrected set and a `gpt-4o` judge. Figures come from single runs, so differences of 1–2 points are within noise.

---

## Observability

**Langfuse (optional).** When Langfuse keys are configured, queries are traced end-to-end: retrieval, classification, reranking and generation, with latency and token usage per step.

**Usage and cost.** Every external model call logs its billable usage (tokens or rerank search units, no content) as a `USAGE` line in the API logs. Totals and estimated cost for a time window:

```bash
python scripts/cost_report.py --since 2026-11-15T10:00:00Z
```

Prices are list prices hard-coded in `scripts/cost_report.py`; verify them against the providers' pricing pages.

---

## Setup

### Prerequisites

- Docker + Docker Compose
- OpenAI API key
- Cohere API key (a trial key works but is limited to 10 rerank calls per minute)
- Langfuse API keys (optional)

### 1. Configure environment

Create a `.env` file in the project root:

```env
POSTGRES_USER=hrcopilot
POSTGRES_PASSWORD=...
POSTGRES_DB=hrcopilot
DATABASE_URL=postgresql+asyncpg://{user}:{password}@db:5432/{db}

OPENAI_API_KEY=sk-...
COHERE_API_KEY=...

# Langfuse observability (optional)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

PGADMIN_DEFAULT_EMAIL=...
PGADMIN_DEFAULT_PASSWORD=...
```

> `.env` is gitignored — never commit it.
> Database and pgAdmin credentials here are for local development only.

### 2. Start services

```bash
docker compose up --build
```

This starts:
- `db` — PostgreSQL 16 with pgvector, published on host port `5433`
- `api` — FastAPI on port `8000` with hot reload
- `pgadmin` — pgAdmin 4 on port `5050`

### 3. Run migrations

```bash
docker compose exec api alembic upgrade head
```

### 4. Verify

- Swagger API docs: `http://localhost:8000/docs`
- pgAdmin: `http://localhost:5050`

---

## API

### Create a tenant

```bash
curl -X POST http://localhost:8000/api/v1/tenants/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp", "slug": "acme-corp"}'
```

### Upload a document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload?tenant_id=<uuid>" \
  -F "file=@/path/to/document.pdf;type=application/pdf"
```

Supported types: PDF and plain text. Documents ingested before section-aware chunking keep their original chunks until they are uploaded again.

### Query

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "<uuid>",
    "question": "What is the notice period for termination?"
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
      "content": "[Employment Agreement — s.12 Termination]\n12. Termination.- ...",
      "rrf_score": 0.032787,
      "relevance_score": 0.891234
    }
  ]
}
```

`rrf_score` — combined rank from vector and keyword fusion (higher = retrieved by more signals).
`relevance_score` — Cohere reranker score (semantic relevance to the question).

---

## Database

pgAdmin is available at `http://localhost:5050`:

| Field | Value |
|---|---|
| Login email | set in `.env` (`PGADMIN_DEFAULT_EMAIL`) |
| Login password | set in `.env` (`PGADMIN_DEFAULT_PASSWORD`) |
| Host | `db` |
| Port | `5432` |
| Database | set in `.env` (`POSTGRES_DB`) |
| Username | set in `.env` (`POSTGRES_USER`) |
| Password | set in `.env` (`POSTGRES_PASSWORD`) |

---

## Evaluation

### 1. Build an eval set (`scripts/build_eval.py`)

Generates questions from the PDFs using the same text extraction as ingestion, so evidence quotes match the indexed text.

- Pages are sampled across each document (start, middle and end), not just the opening pages.
- Each question must name its document; boilerplate (short title, commencement, gazette details) is excluded.
- Each answer carries a verbatim evidence quote and page number.
- Questions are validated before being kept: vague-reference patterns, evidence quote present in the source, and an LLM check that the question is standalone, supported and substantive.
- Unanswerable questions are generated as out-of-corpus questions and as near-misses (a real document, asked about a detail it does not contain).
- A Markdown review file is written for human spot-checking, along with a manifest of documents and their extracted titles.

```bash
# Pilot on a few documents, review, then generate for the full corpus
python scripts/build_eval.py --files a.pdf b.pdf --out scripts/eval_sets/pilot.json
python scripts/build_eval.py --all --out-of-corpus 10 --near-miss 8 --out scripts/eval_sets/legal_v2.json
```

The current benchmark is `scripts/eval_sets/legal_v2.json`. Entries corrected after manual triage carry a `fix_note`.

### 2. Run the eval (`scripts/run_eval.py`)

Queries the running API, scores retrieval and answers, and writes a text report plus raw JSON results to `eval_results/`.

```bash
python scripts/run_eval.py \
  --tenant-id <uuid> \
  --eval-file scripts/eval_sets/legal_v2.json \
  --judge-model gpt-4o \
  --workers 2
```

**Scoring:**
- **Correct document / evidence in sources** — whether the expected document, and the evidence passage itself, appear in the top retrieved sources. The legacy "retrieval accuracy" line requires an exact page match and understates section-level chunks, which start on the page where the section begins.
- **Answer accuracy** — an LLM judge compares the answer with the expected answer, given the evidence quote and all retrieved sources. Answers that cite provisions found in neither are marked as failures.
- **Escalations and refusals** — counted separately, so classifier errors can be told apart from retrieval errors.
- **Unanswerable questions** (`should_retrieve: false`) — pass only if the system declines to answer.
- **Cost** — each report ends with the token usage and estimated cost of the run, split between the system under test and the judge.

With a Cohere trial key, keep `--workers` at 2: reranking is limited to 10 calls per minute, and a full run takes about 18 minutes.

### 3. Direct pipeline evaluation (`eval_runner.py`)

Runs the query service directly against the database session, without network calls to the API:

```bash
docker compose exec api python eval_runner.py --tenant-id <uuid>
```

---

## Project Structure

```
hr-copilot/
├── app/
│   ├── agent/                   # LangGraph agent implementation
│   │   ├── graph.py             # StateGraph definition & routing
│   │   ├── nodes.py             # retrieve, classify, answer, escalate nodes
│   │   └── state.py             # AgentState schema
│   ├── api/v1/endpoints/        # tenants, documents, query
│   ├── core/config.py           # Pydantic settings & env loading
│   ├── db/session.py            # Async SQLAlchemy session engine
│   ├── models/                  # Tenant, Document, Chunk (pgvector + tsvector)
│   ├── schemas/                 # Pydantic request/response models
│   └── services/
│       ├── parsing.py           # PDF extraction, section-aware chunking
│       ├── document_service.py  # Ingestion: titles, chunk headers, embeddings
│       ├── embedding_service.py # OpenAI embeddings
│       ├── bm25_service.py      # Postgres full-text keyword search
│       ├── rrf.py               # Reciprocal Rank Fusion
│       ├── reranker_service.py  # Cohere reranker with rate-limit retries
│       ├── usage.py             # Model-call usage logging
│       └── query_service.py     # Direct query orchestration
├── alembic/                     # DB migrations
├── eval.json                    # HR baseline eval set (10 questions)
├── eval_results/                # Eval reports (gitignored)
├── eval_runner.py               # Direct DB eval runner
├── scripts/
│   ├── build_eval.py            # Eval set generator
│   ├── run_eval.py              # Live API eval runner
│   ├── cost_report.py           # Usage and cost totals from API logs
│   ├── eval_sets/               # Generated eval sets, manifests, review files
│   ├── legal_eval.json          # Legacy legal Q&A set (superseded by eval_sets/legal_v2.json)
│   ├── run_eval_3.json          # Legacy 3-document benchmark
│   ├── download_50_pdfs.py      # PDF dataset scraper
│   └── ingest_test_pdfs.py      # Batch ingestion (--tenant-id, --pdf-dir)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
