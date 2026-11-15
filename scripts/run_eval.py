#!/usr/bin/env python3
"""
Simple evaluation runner for HR Copilot.

Queries the FastAPI /api/v1/query/ endpoint and evaluates retrieval and answer
accuracy against ground truth test cases (e.g. legal_eval.json).

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --limit 10
    python scripts/run_eval.py --tenant-id <uuid> --eval-file scripts/legal_eval.json
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime, timezone
import difflib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import dotenv
from openai import OpenAI
import requests

from cost_report import print_table

# Load .env from repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(REPO_ROOT / ".env")

# Default configuration (matches scripts/ingest_test_pdfs.py)
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TENANT_ID = "8fa00d66-ad8d-44cc-baf6-a71c7540dc84"
DEFAULT_EVAL_FILE = Path(__file__).parent / "legal_eval.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_results"

JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for an authoritative RAG system.
You will be given:
1. A Question
2. An Expected Answer (reference ground truth)
3. Source Evidence (the passage from the source document the expected answer is based on, if available)
4. Retrieved Context (document excerpts retrieved by the system, if available)
5. The Actual Answer produced by the system

Your job is to decide if the actual answer is factually correct and answers the question accurately.

Rules:
- Minor wording differences are fine — evaluate semantic meaning and factual truth.
- If expected_answer is "Not explicitly mentioned" and the actual answer also states it does not have the information / not found in the context, that is a PASS.
- Comprehensiveness & Elaboration: If the actual answer covers the core facts of the expected answer and includes additional accurate details, qualifications, or legal provisions supported by the retrieved context, that is a PASS. Do NOT penalize the system for being more complete, detailed, or thorough than the expected reference.
- Mark FAIL only if:
  1. The actual answer directly contradicts the expected answer or asserts factually false information.
  2. The actual answer misses the core question being asked.
  3. For unanswerable questions ("Not explicitly mentioned"), the actual answer hallucinates/makes up an answer instead of admitting it lacks information.
  4. The actual answer cites laws, provisions or facts that appear in neither the Source Evidence nor the Retrieved Context.

Respond with ONLY a JSON object: {"verdict": "PASS" or "FAIL", "reason": "<one sentence explanation>"}"""


def score_retrieval(expected_doc: str | None, expected_page: int | None, sources: list[dict]) -> bool | None:
    """Check if expected document (matching stem/name regardless of .txt vs .pdf) was retrieved."""
    if not expected_doc:
        return None

    expected_stem = Path(expected_doc).stem.lower()

    for s in sources:
        doc_title = s.get("document_title", "")
        doc_stem = Path(doc_title).stem.lower()

        # Match by stem (e.g. 6976fbb07496b_788 matches .txt or .pdf) or substring
        if expected_stem == doc_stem or expected_stem in doc_stem or doc_stem in expected_stem:
            if expected_page is not None:
                if s.get("page_number") == expected_page:
                    return True
            else:
                return True
    return False


JUDGE_USAGE: list[tuple[str, int, int]] = []  # (model, input tokens, output tokens) per judge call


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def score_quote(evidence_quote: str | None, sources: list[dict]) -> bool | None:
    """Check if the evidence passage appears (allowing OCR/chunk-boundary slack) in any retrieved chunk."""
    if not evidence_quote:
        return None
    q = _norm(evidence_quote)
    for s in sources:
        t = _norm(s.get("content", ""))
        if q in t:
            return True
        m = difflib.SequenceMatcher(None, t, q, autojunk=False).find_longest_match(0, len(t), 0, len(q))
        # Chunk boundaries can cut a long quote, so a solid partial overlap counts
        if m.size >= min(0.6 * len(q), 150):
            return True
    return False


def is_refusal(answer: str) -> bool:
    return answer.startswith("I wasn't able to find") or "don't have that information" in answer


def judge_answer(
    client: OpenAI,
    question: str,
    expected: str,
    actual: str,
    sources: list[dict] | None = None,
    evidence_quote: str | None = None,
    model: str = "gpt-4o-mini",
) -> tuple[str, str]:
    """Judge answer correctness against the expected answer, source evidence and retrieved context."""
    evidence_text = f"\n--- SOURCE EVIDENCE ---\n{evidence_quote}\n--- END EVIDENCE ---\n" if evidence_quote else ""
    context_text = ""
    if sources:
        snippets = [s.get("content", "").strip() for s in sources if s.get("content", "").strip()]
        if snippets:
            context_text = "\n--- RETRIEVED CONTEXT ---\n" + "\n\n".join(snippets) + "\n--- END CONTEXT ---\n\n"

    user_msg = (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"{evidence_text}"
        f"{context_text}"
        f"Actual answer: {actual}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        JUDGE_USAGE.append((model, resp.usage.prompt_tokens, resp.usage.completion_tokens))
        parsed = json.loads(resp.choices[0].message.content)
        return parsed.get("verdict", "FAIL"), parsed.get("reason", "")
    except Exception as e:
        return "ERROR", str(e)


def build_report_text(results: list[dict]) -> str:
    """Format evaluation results matching the eval_results/10-docs.txt style."""
    total = len(results)
    retrieval_cases = [r for r in results if r["should_retrieve"]]
    negative_cases = [r for r in results if not r["should_retrieve"]]

    retrieval_passes = sum(1 for r in retrieval_cases if r["retrieval_pass"])
    answer_passes = sum(1 for r in results if r["answer_verdict"] == "PASS")

    # False positive: negative unanswerable case where the model failed to refuse and attempted/hallucinated an answer
    false_positives = [
        r for r in negative_cases
        if r["answer_verdict"] != "PASS"
    ]

    ret_acc_pct = int(round((retrieval_passes / len(retrieval_cases) * 100))) if retrieval_cases else 0
    ans_acc_pct = int(round((answer_passes / total * 100))) if total else 0

    def pct(n: int, d: int) -> str:
        return f"{n}/{d} ({round(n / d * 100) if d else 0}%)"

    doc_passes = sum(1 for r in retrieval_cases if r.get("doc_pass"))
    quote_cases = [r for r in retrieval_cases if r.get("quote_pass") is not None]
    quote_passes = sum(1 for r in quote_cases if r["quote_pass"])
    escalated = sum(1 for r in retrieval_cases if r.get("escalated"))
    refused = sum(1 for r in retrieval_cases if r.get("refused"))

    lines = [
        "=" * 70,
        "  HR COPILOT EVAL REPORT",
        "=" * 70,
        "",
        "SUMMARY",
        f"  Total questions       : {total}",
        f"  Should-retrieve cases : {len(retrieval_cases)}",
        f"  Negative cases        : {len(negative_cases)}",
        "",
        f"  Retrieval accuracy    : {retrieval_passes}/{len(retrieval_cases)} ({ret_acc_pct}%)",
        f"  Answer accuracy       : {answer_passes}/{total} ({ans_acc_pct}%)",
        f"  False positives       : {len(false_positives)}/{len(negative_cases)}",
        "",
        "  Positive cases, final sources (escalations return no sources):",
        f"    Right doc in sources  : {pct(doc_passes, len(retrieval_cases))}",
        f"    Evidence in sources   : {pct(quote_passes, len(quote_cases))}",
        f"    Escalated (classifier): {pct(escalated, len(retrieval_cases))}",
        f"    Refused (any reason)  : {pct(refused, len(retrieval_cases))}",
        "",
        "-" * 70,
        "  PER-QUESTION BREAKDOWN",
        "-" * 70,
    ]

    for i, r in enumerate(results, 1):
        if r["should_retrieve"]:
            ret_label = "RETRIEVE ✓" if r["retrieval_pass"] else "RETRIEVE ✗"
        else:
            is_fp = r["answer_verdict"] != "PASS"
            ret_label = "NEG CASE" + (" [FP!]" if is_fp else " ✓")

        ans_symbol = "✓" if r["answer_verdict"] == "PASS" else ("?" if r["answer_verdict"] == "ERROR" else "✗")

        lines.append(f"\n  Q{i}: {r['question']}")
        lines.append(f"       Retrieval : {ret_label}")
        lines.append(f"       Answer    : {r['answer_verdict']} {ans_symbol}  — {r['answer_reason']}")
        lines.append(f"       Expected  : {r['expected_answer']}")
        actual_snippet = r['actual_answer'][:200] + ("..." if len(r['actual_answer']) > 200 else "")
        lines.append(f"       Actual    : {actual_snippet}")

        if r["should_retrieve"] and not r["retrieval_pass"]:
            lines.append(f"       Expected doc  : {r['expected_doc']}")
            retrieved_summary = [
                f"{doc} (p.{pg})" if pg else doc
                for doc, pg in zip(r["retrieved_docs"], r["retrieved_pages"])
            ]
            lines.append(f"       Retrieved docs: {retrieved_summary}")

        if r.get("error"):
            lines.append(f"       ERROR     : {r['error']}")

    lines.append("\n" + "=" * 70 + "\n")
    return "\n".join(lines)


def cost_text(since: str) -> str:
    """Cost of this run: the system's model calls (from the API's USAGE logs) plus the judge's."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("COST (list prices, USD — verify against provider pricing)")
        print(" System under test:")
        try:
            logs = subprocess.run(
                ["docker", "compose", "logs", "api", "--no-log-prefix", "--since", since],
                capture_output=True, text=True, check=True, cwd=REPO_ROOT,
            ).stdout
            totals: dict = defaultdict(lambda: defaultdict(int))
            for line in logs.splitlines():
                if line.startswith("USAGE "):
                    u = json.loads(line[len("USAGE "):])
                    t = totals[u["model"]]
                    t["calls"] += 1
                    for k in ("input_tokens", "output_tokens", "search_units"):
                        t[k] += u[k]
            system = print_table(totals)
        except Exception as e:
            print(f"  I don't know (could not read API logs: {e})")
            system = None
        print(" Judge:")
        judge: dict = defaultdict(lambda: defaultdict(int))
        for model, tin, tout in JUDGE_USAGE:
            t = judge[model]
            t["calls"] += 1
            t["input_tokens"] += tin
            t["output_tokens"] += tout
            t["search_units"] += 0
        judge_cost = print_table(judge)
        if system is not None:
            print(f" Run total: ${system + judge_cost:.4f}")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="HR Copilot simple evaluation runner")
    parser.add_argument("--tenant-id", default=DEFAULT_TENANT_ID, help="Tenant UUID")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base API URL (default: http://localhost:8000)")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE, help="Path to evaluation JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    parser.add_argument("--output-file", type=Path, default=None, help="Custom path to save text report")
    parser.add_argument("--workers", type=int, default=4, help="Questions to run in parallel")
    parser.add_argument("--judge-model", default="gpt-4o", help="Model that grades answers")
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Do not filter test cases to only documents ingested in the tenant",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not found. Please check your .env file.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    if not args.eval_file.exists():
        print(f"Error: Eval file not found: {args.eval_file}", file=sys.stderr)
        sys.exit(1)

    # Test server health before starting
    try:
        health_resp = requests.get(f"{args.api_url.rstrip('/')}/health", timeout=5.0)
        if health_resp.status_code != 200:
            print(f"Warning: /health returned HTTP {health_resp.status_code}")
    except requests.RequestException:
        print(
            f"Error: Could not connect to FastAPI server at {args.api_url}.\n"
            f"Please ensure your server is running (e.g. uvicorn app.main:app --reload).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fetch ingested documents for tenant to filter test cases
    ingested_titles = []
    try:
        docs_resp = requests.get(
            f"{args.api_url.rstrip('/')}/api/v1/documents/?tenant_id={args.tenant_id}",
            timeout=10.0,
        )
        if docs_resp.status_code == 200:
            ingested_titles = [d.get("title", "") for d in docs_resp.json()]
    except Exception as e:
        print(f"Note: Could not query tenant documents: {e}")

    ingested_stems = {Path(t).stem.lower() for t in ingested_titles if t}

    raw_cases = json.loads(args.eval_file.read_text(encoding="utf-8"))

    if not args.no_filter and ingested_stems:
        filtered = []
        for c in raw_cases:
            if not c.get("should_retrieve", True):
                filtered.append(c)  # Keep all unanswerable negative cases
            else:
                doc = c.get("expected_source_document")
                stem = Path(doc).stem.lower() if doc else ""
                if stem in ingested_stems:
                    filtered.append(c)

        skipped = len(raw_cases) - len(filtered)
        pos_count = len(filtered) - sum(1 for c in filtered if not c.get("should_retrieve", True))
        neg_count = sum(1 for c in filtered if not c.get("should_retrieve", True))
        print(f"Filtered eval set to documents ingested in DB ({len(ingested_stems)} docs in tenant):")
        print(f"  -> Kept {len(filtered)} cases ({pos_count} positive + {neg_count} negative unanswerable)")
        if skipped > 0:
            print(f"  -> Skipped {skipped} cases whose documents are not in this tenant.")
        raw_cases = filtered

    if args.limit:
        raw_cases = raw_cases[: args.limit]

    print("=" * 70)
    print(f"Starting Evaluation on {len(raw_cases)} questions")
    print(f"Target Tenant : {args.tenant_id}")
    print(f"API Endpoint  : {args.api_url}/api/v1/query/")
    print(f"Eval File     : {args.eval_file}")
    print("=" * 70 + "\n")

    query_endpoint = f"{args.api_url.rstrip('/')}/api/v1/query/"

    def run_case(i: int, case: dict) -> dict:
        q = case["question"]
        expected_ans = case["expected_answer"]
        expected_doc = case.get("expected_source_document")
        expected_page = case.get("expected_page")
        should_retrieve = case.get("should_retrieve", True)
        meta = {"id": case.get("id"), "category": case.get("category"), "evidence_quote": case.get("evidence_quote")}

        print(f"[{i:03d}/{len(raw_cases)}] Querying: {q[:65]}...")

        try:
            resp = requests.post(
                query_endpoint,
                json={
                    "tenant_id": args.tenant_id,
                    "question": q,
                },
                timeout=90.0,
            )

            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:100]}")

            data = resp.json()
            actual_ans = data.get("answer", "")
            sources = data.get("sources", [])

            retrieved_docs = [s.get("document_title", "") for s in sources]
            retrieved_pages = [s.get("page_number") for s in sources]

            retrieval_pass = score_retrieval(expected_doc, expected_page, sources)
            verdict, reason = judge_answer(
                client, q, expected_ans, actual_ans, sources=sources,
                evidence_quote=case.get("evidence_quote"), model=args.judge_model,
            )

            return {
                **meta,
                "doc_pass": score_retrieval(expected_doc, None, sources),
                "quote_pass": score_quote(case.get("evidence_quote"), sources),
                "escalated": actual_ans.startswith("I wasn't able to find"),
                "refused": is_refusal(actual_ans),
                "question": q,
                "expected_answer": expected_ans,
                "expected_doc": expected_doc,
                "expected_page": expected_page,
                "should_retrieve": should_retrieve,
                "actual_answer": actual_ans,
                "sources": sources,
                "retrieved_docs": retrieved_docs,
                "retrieved_pages": retrieved_pages,
                "retrieval_pass": retrieval_pass,
                "answer_verdict": verdict,
                "answer_reason": reason,
                "error": None,
            }

        except Exception as e:
            return {
                **meta,
                "question": q,
                "expected_answer": expected_ans,
                "expected_doc": expected_doc,
                "expected_page": expected_page,
                "should_retrieve": should_retrieve,
                "actual_answer": "",
                "sources": [],
                "retrieved_docs": [],
                "retrieved_pages": [],
                "retrieval_pass": False if should_retrieve else None,
                "answer_verdict": "ERROR",
                "answer_reason": "",
                "error": str(e),
            }

    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_case, range(1, len(raw_cases) + 1), raw_cases))

    # Generate and print report
    report_text = build_report_text(results) + "\n" + cost_text(started)
    print(report_text)

    # Save report
    out_path = args.output_file
    if not out_path:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = DEFAULT_OUTPUT_DIR / f"eval_report_{args.eval_file.stem}_{ts}.txt"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(report_text, encoding="utf-8")
    out_path.with_suffix(".json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report saved to: {out_path} (raw results: {out_path.with_suffix('.json').name})\n")


if __name__ == "__main__":
    main()
