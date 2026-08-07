"""
Eval runner for HR Copilot RAG pipeline.

Usage:
    python eval_runner.py --tenant-id <uuid>
    python eval_runner.py --tenant-id <uuid> --eval-file path/to/eval.json
"""
import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.services.query_service import run_query


JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for a RAG system.
You will be given a question, an expected answer, and the actual answer produced by the system.
Your job is to decide if the actual answer is semantically equivalent to or a correct answer for the question,
given the expected answer as a reference.

Rules:
- Minor wording differences are fine — judge meaning, not exact match.
- If expected_answer is "Not explicitly mentioned" and the actual answer also says it doesn't know / isn't in the context, that's a PASS.
- If the actual answer adds correct extra detail beyond the expected, that's still a PASS.
- If the actual answer is factually wrong or contradicts the expected, that's a FAIL.
- If the actual answer hallucinated information not in the expected, that's a FAIL.

Respond with ONLY a JSON object: {"verdict": "PASS" or "FAIL", "reason": "<one sentence>"}"""


@dataclass
class EvalCase:
    question: str
    expected_answer: str
    expected_source_document: str | None
    expected_page: int | None
    should_retrieve: bool


@dataclass
class EvalResult:
    case: EvalCase
    retrieval_pass: bool | None  # None = not applicable (should_retrieve=False)
    answer_verdict: str          # PASS / FAIL / ERROR
    answer_reason: str
    actual_answer: str
    retrieved_docs: list[str]    # document titles from sources
    retrieved_pages: list[int | None]
    error: str | None = None


def score_retrieval(case: EvalCase, sources: list) -> bool:
    """True if expected document (and page, when specified) appears in sources."""
    for src in sources:
        title_match = src.document_title == case.expected_source_document
        if case.expected_page is not None:
            if title_match and src.page_number == case.expected_page:
                return True
        else:
            if title_match:
                return True
    return False


async def judge_answer(
    client: AsyncOpenAI,
    question: str,
    expected: str,
    actual: str,
) -> tuple[str, str]:
    """Returns (verdict, reason). verdict is PASS, FAIL, or ERROR."""
    user_msg = (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Actual answer: {actual}"
    )
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        return parsed.get("verdict", "ERROR"), parsed.get("reason", "")
    except Exception as e:
        return "ERROR", str(e)


def print_report(results: list[EvalResult]) -> None:
    total = len(results)
    retrieval_cases = [r for r in results if r.case.should_retrieve]
    negative_cases = [r for r in results if not r.case.should_retrieve]

    retrieval_passes = sum(1 for r in retrieval_cases if r.retrieval_pass)
    answer_passes = sum(1 for r in results if r.answer_verdict == "PASS")
    answer_errors = sum(1 for r in results if r.answer_verdict == "ERROR")

    # negative cases: should NOT retrieve but did retrieve
    false_positives = [
        r for r in negative_cases
        if r.case.expected_source_document is not None
        and any(r.case.expected_source_document in doc for doc in r.retrieved_docs)
    ]

    print("\n" + "=" * 70)
    print("  HR COPILOT EVAL REPORT")
    print("=" * 70)

    print(f"\n{'SUMMARY':}")
    print(f"  Total questions       : {total}")
    print(f"  Should-retrieve cases : {len(retrieval_cases)}")
    print(f"  Negative cases        : {len(negative_cases)}")
    print()
    print(f"  Retrieval accuracy    : {retrieval_passes}/{len(retrieval_cases)} "
          f"({retrieval_passes/len(retrieval_cases)*100:.0f}%)" if retrieval_cases else "  Retrieval accuracy    : n/a")
    print(f"  Answer accuracy       : {answer_passes}/{total} "
          f"({answer_passes/total*100:.0f}%)"
          + (f"  [{answer_errors} ERROR(s)]" if answer_errors else ""))
    print(f"  False positives       : {len(false_positives)}/{len(negative_cases)}")

    print("\n" + "-" * 70)
    print("  PER-QUESTION BREAKDOWN")
    print("-" * 70)

    for i, r in enumerate(results, 1):
        ret_label = ""
        if r.case.should_retrieve:
            ret_label = "RETRIEVE ✓" if r.retrieval_pass else "RETRIEVE ✗"
        else:
            fp = (
                r.case.expected_source_document is not None
                and any(r.case.expected_source_document in doc for doc in r.retrieved_docs)
            )
            ret_label = "NEG CASE" + (" [FP!]" if fp else " ✓")

        ans_symbol = "✓" if r.answer_verdict == "PASS" else ("?" if r.answer_verdict == "ERROR" else "✗")
        print(f"\n  Q{i}: {r.case.question}")
        print(f"       Retrieval : {ret_label}")
        print(f"       Answer    : {r.answer_verdict} {ans_symbol}  — {r.answer_reason}")
        print(f"       Expected  : {r.case.expected_answer}")
        print(f"       Actual    : {r.actual_answer[:200]}{'...' if len(r.actual_answer) > 200 else ''}")

        if r.case.should_retrieve and not r.retrieval_pass:
            print(f"       Expected doc  : {r.case.expected_source_document} p.{r.case.expected_page}")
            retrieved = [
                f"{doc} p.{pg}" for doc, pg in zip(r.retrieved_docs, r.retrieved_pages)
            ]
            print(f"       Retrieved docs: {retrieved}")

        if r.error:
            print(f"       ERROR: {r.error}")

    print("\n" + "=" * 70 + "\n")


from datetime import datetime


def save_results(
    results: list[EvalResult],
    eval_file: Path,
    tenant_id: uuid.UUID,
    output_path: Path | None = None,
) -> Path:
    out_dir = Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = out_dir / f"results_{eval_file.stem}_{timestamp}.json"

    retrieval_cases = [r for r in results if r.case.should_retrieve]
    retrieval_passes = sum(1 for r in retrieval_cases if r.retrieval_pass)
    answer_passes = sum(1 for r in results if r.answer_verdict == "PASS")

    serialized_results = []
    for r in results:
        serialized_results.append({
            "question": r.case.question,
            "expected_answer": r.case.expected_answer,
            "actual_answer": r.actual_answer,
            "should_retrieve": r.case.should_retrieve,
            "expected_doc": r.case.expected_source_document,
            "expected_page": r.case.expected_page,
            "retrieved_docs": r.retrieved_docs,
            "retrieved_pages": r.retrieved_pages,
            "retrieval_pass": r.retrieval_pass,
            "answer_verdict": r.answer_verdict,
            "answer_reason": r.answer_reason,
            "error": r.error,
        })

    payload = {
        "tenant_id": str(tenant_id),
        "eval_file": str(eval_file),
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_questions": len(results),
            "should_retrieve_cases": len(retrieval_cases),
            "retrieval_passes": retrieval_passes,
            "retrieval_accuracy": (
                retrieval_passes / len(retrieval_cases) if retrieval_cases else None
            ),
            "answer_passes": answer_passes,
            "answer_accuracy": answer_passes / len(results) if results else 0,
        },
        "results": serialized_results,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[Saved eval results to {output_path}]")
    return output_path


async def run_eval(
    tenant_id: uuid.UUID, eval_file: Path, output_file: Path | None = None
) -> None:
    raw = json.loads(eval_file.read_text())
    cases = [EvalCase(**c) for c in raw]

    judge_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    results: list[EvalResult] = []

    async with AsyncSessionLocal() as db:
        for i, case in enumerate(cases, 1):
            print(f"Running Q{i}/{len(cases)}: {case.question[:60]}...")
            try:
                response = await run_query(
                    db=db,
                    tenant_id=tenant_id,
                    question=case.question,
                )

                retrieved_docs = [s.document_title for s in response.sources]
                retrieved_pages = [s.page_number for s in response.sources]

                retrieval_pass = None
                if case.should_retrieve:
                    retrieval_pass = score_retrieval(case, response.sources)

                verdict, reason = await judge_answer(
                    client=judge_client,
                    question=case.question,
                    expected=case.expected_answer,
                    actual=response.answer,
                )

                results.append(EvalResult(
                    case=case,
                    retrieval_pass=retrieval_pass,
                    answer_verdict=verdict,
                    answer_reason=reason,
                    actual_answer=response.answer,
                    retrieved_docs=retrieved_docs,
                    retrieved_pages=retrieved_pages,
                ))

            except Exception as e:
                results.append(EvalResult(
                    case=case,
                    retrieval_pass=False,
                    answer_verdict="ERROR",
                    answer_reason="",
                    actual_answer="",
                    retrieved_docs=[],
                    retrieved_pages=[],
                    error=str(e),
                ))

    print_report(results)
    save_results(results, eval_file, tenant_id, output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG eval suite")
    parser.add_argument("--tenant-id", required=True, help="Tenant UUID to query against")
    parser.add_argument(
        "--eval-file",
        default="eval.json",
        help="Path to eval JSON file (default: eval.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional custom path to save JSON results (default: eval_results/results_<eval_file>_<timestamp>.json)",
    )
    args = parser.parse_args()

    try:
        tenant_id = uuid.UUID(args.tenant_id)
    except ValueError:
        print(f"Invalid tenant-id: {args.tenant_id}")
        sys.exit(1)

    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        print(f"Eval file not found: {eval_file}")
        sys.exit(1)

    output_file = Path(args.output) if args.output else None
    asyncio.run(run_eval(tenant_id, eval_file, output_file))


if __name__ == "__main__":
    main()
