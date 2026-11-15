#!/usr/bin/env python3
"""
Build a retrieval eval set from the PDFs in test-data.

Text is extracted with the app's own parser, so evidence quotes match what gets
ingested. Questions are sampled across each document (not just page 1), must name
their document, and are validated before being kept.

Usage:
    python scripts/build_eval.py --files a.pdf b.pdf --out scripts/eval_sets/legal_pilot.json
    python scripts/build_eval.py --all --out scripts/eval_sets/legal_v2.json
"""
import argparse
import asyncio
import difflib
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import dotenv
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
dotenv.load_dotenv(REPO_ROOT / ".env")

from app.services.parsing import extract_text_by_page  # noqa: E402

PDF_DIR = Path("/mnt/c/Users/khiza/Documents/test-data")
MIN_PAGE_CHARS = 400
CATEGORIES = ["lookup", "numeric", "procedural", "definition", "plain_language"]
NOT_FOUND = "Not explicitly mentioned in the context."

# Phrases that make a question unanswerable without the source open
VAGUE = re.compile(
    r"\b(this|the said|the above) (act|bill|ordinance|document|law|amendment)\b"
    r"|\bthe (document|excerpt|text|passage|page)\b"
    r"|\bmentioned in\b|\baccording to the amendments?\b",
    re.I,
)

TITLE_PROMPT = """Below is the opening text of a Pakistani legal document (OCR, may contain errors).
Return its official title, e.g. "The Domestic Violence (Prevention and Protection) Act, 2026".
For an Act/Bill/Ordinance use its short title. Otherwise use the document's own title.
Correct obvious OCR errors in the title.

<<<
{text}
>>>

Respond with JSON: {{"title": "..."}}"""

GEN_PROMPT = """You are writing questions for a retrieval benchmark over ~50 Pakistani legal documents.
The question will be asked to a search system by someone who has NOT seen this page.

Document: {title}
Page {page} of {pages} (OCR text, may contain errors):
<<<
{text}
>>>

Write ONE question answerable from this page.
Rules:
- Name the document in the question ("Under the {title}, ..."). Never say "this Act", "the document",
  "the excerpt", or "the amendment" without naming it.
- Do NOT ask about the short title, commencement, gazette publication, assent date, or signatories.
- Ask about substance: duties, powers, conditions, penalties, time limits, amounts, definitions,
  procedures, or (for rulings) what the Chair ruled and why.
- The answer must be fully supported by this page.
- evidence_quote: copy ONE contiguous passage (max 300 chars) VERBATIM from the page text above,
  keeping any OCR errors, that supports the answer.
- category: one of {categories}. Prefer "{hint}" if the page allows it.
  "plain_language" means phrased the way a layperson would ask, without legal jargon (still name the document).

Respond with JSON: {{"question": "...", "expected_answer": "...", "evidence_quote": "...", "category": "..."}}"""

CHECK_PROMPT = """You are reviewing one question from a retrieval benchmark over ~50 Pakistani legal documents.

Source document: {title}, page {page}
<<<
{text}
>>>

Question: {question}
Expected answer: {answer}

Check:
1. standalone: does the QUESTION itself identify which document it is about? Naming the document's title
   (e.g. "Under the {title}, ...") is enough. False only if it relies on unnamed references such as
   "this Act", "the amendment", "the document". Judge the question only, not the answer.
2. supported: is the expected answer correct and fully supported by the page?
3. substantive: is it about substance rather than short title / commencement / publication boilerplate?

Respond with JSON: {{"standalone": true/false, "supported": true/false, "substantive": true/false, "reason": "<one sentence>"}}"""

OUT_OF_CORPUS_PROMPT = """These documents are in a Pakistani legal search corpus:
{titles}

Write {count} realistic questions about Pakistani law that NONE of these documents could answer.
Make them plausible (they should sound like they could be in such a corpus), not exotic.

Respond with JSON: {{"questions": ["...", "..."]}}"""

NEAR_MISS_PROMPT = """Below is the full text of "{title}" (OCR, may contain errors).
<<<
{text}
>>>

Write ONE question that names this document and asks about a plausible detail the text does NOT contain
(e.g. a penalty, deadline, or authority that it never specifies). Someone skimming might expect it to be there.
Do not hint in the question that the detail is missing.

Respond with JSON: {{"question": "..."}}"""


async def ask(client: AsyncOpenAI, model: str, prompt: str, temperature: float = 0.0) -> dict:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_found(quote: str, text: str) -> bool:
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    m = difflib.SequenceMatcher(None, t, q, autojunk=False).find_longest_match(0, len(t), 0, len(q))
    return m.size >= 0.8 * len(q)


def questions_for(page_count: int) -> int:
    return min(8, max(2, math.ceil(page_count / 6) + 1))


def sample_pages(pages: list[tuple[int, str]], n: int, seed: str) -> list[int]:
    """Spread n picks evenly across the document, skipping near-empty and cover pages."""
    eligible = [p for p, t in pages if len(t) >= MIN_PAGE_CHARS]
    if len(pages) > 2:
        eligible = [p for p in eligible if p != 1] or eligible
    if not eligible:
        return []
    rng = random.Random(seed)
    n = min(n, len(eligible))
    buckets = [eligible[i * len(eligible) // n:(i + 1) * len(eligible) // n] for i in range(n)]
    return [rng.choice(b) for b in buckets if b]


async def build_doc(client, model, sem, path: Path) -> tuple[dict, list[dict], list[dict]]:
    pages = extract_text_by_page(path.read_bytes())
    text_by_page = dict(pages)
    async with sem:
        title = (await ask(client, model, TITLE_PROMPT.format(text=" ".join(t for _, t in pages[:2])[:3000])))["title"]
    doc = {"file": path.name, "title": title, "pages": len(pages), "text": " ".join(t for _, t in pages)}

    picks = sample_pages(pages, questions_for(len(pages)), seed=path.name)
    rng = random.Random(path.name)

    async def one(page: int) -> tuple[dict | None, dict | None]:
        # Include the start of the next page so sections that span a page break stay intact
        context = text_by_page[page] + " " + text_by_page.get(page + 1, "")[:600]
        async with sem:
            item = await ask(client, model, GEN_PROMPT.format(
                title=title, page=page, pages=len(pages), text=context,
                categories=", ".join(CATEGORIES), hint=rng.choice(CATEGORIES),
            ), temperature=0.3)
        q, a, quote = item.get("question", ""), item.get("expected_answer", ""), item.get("evidence_quote", "")
        reasons = []
        if VAGUE.search(q):
            reasons.append("vague reference")
        # The quote may come from the next-page overflow; point expected_page at where it actually is
        if quote_found(quote, text_by_page[page]):
            source_page = page
        elif quote_found(quote, text_by_page.get(page + 1, "")):
            source_page = page + 1
        else:
            source_page = page
            reasons.append("evidence quote not in source")
        if not reasons:
            async with sem:
                check = await ask(client, model, CHECK_PROMPT.format(
                    title=title, page=page, text=context, question=q, answer=a))
            for key in ("standalone", "supported", "substantive"):
                if not check.get(key):
                    reasons.append(f"not {key}: {check.get('reason')}")
        record = {
            "id": f"{path.stem}-p{page}",
            "question": q,
            "expected_answer": a,
            "evidence_quote": quote,
            "expected_source_document": path.name,
            "expected_page": source_page,
            "act_title": title,
            "category": item.get("category"),
            "should_retrieve": True,
        }
        return (None, {**record, "rejected": reasons}) if reasons else (record, None)

    results = await asyncio.gather(*(one(p) for p in picks))
    kept, rejected = [], [r for _, r in results if r]
    for k in (k for k, _ in results if k):
        # Neighbouring picks can land on the same passage
        if any(quote_found(k["evidence_quote"], o["evidence_quote"]) or quote_found(o["evidence_quote"], k["evidence_quote"])
               for o in kept):
            rejected.append({**k, "rejected": ["duplicate of another question's passage"]})
        else:
            kept.append(k)
    return doc, kept, rejected


async def build_negatives(client, model, docs: list[dict], out_of_corpus: int, near_miss: int) -> list[dict]:
    titles = "\n".join(f"- {d['title']}" for d in docs)
    parsed = await ask(client, model, OUT_OF_CORPUS_PROMPT.format(titles=titles, count=out_of_corpus), 0.5)
    negatives = [
        {"id": f"neg-ooc-{i}", "question": q, "category": "negative_out_of_corpus"}
        for i, q in enumerate(parsed.get("questions", [])[:out_of_corpus], 1)
    ]
    # Near-misses only from short documents, so the "not in the text" claim is checkable by a reviewer
    short = [d for d in docs if d["pages"] <= 3][:near_miss]
    for i, d in enumerate(short, 1):
        q = (await ask(client, model, NEAR_MISS_PROMPT.format(title=d["title"], text=d["text"][:8000]), 0.5))["question"]
        negatives.append({"id": f"neg-near-{i}", "question": q, "category": "negative_near_miss",
                          "act_title": d["title"], "near_source": d["file"]})
    return [
        {**n, "expected_answer": NOT_FOUND, "evidence_quote": None,
         "expected_source_document": None, "expected_page": None, "should_retrieve": False}
        for n in negatives
    ]


def write_review(path: Path, kept: list[dict], rejected: list[dict], docs: list[dict]) -> None:
    lines = ["# Eval spot-check", "", f"{len(kept)} kept, {len(rejected)} rejected by the validators.", "",
             "For each: is the question clear and names its document? Is the answer right? Is the quote real?", ""]
    lines += ["## Documents", ""] + [f"- `{d['file']}` ({d['pages']}p): {d['title']}" for d in docs] + [""]
    lines += ["## Questions", ""]
    for i, k in enumerate(kept, 1):
        src = f"`{k['expected_source_document']}` p.{k['expected_page']}" if k["should_retrieve"] else \
            f"negative ({k['category']}{', about ' + k['act_title'] if k.get('act_title') else ''})"
        lines += [f"### {i}. [{k['category']}] {k['question']}", f"- **Answer:** {k['expected_answer']}",
                  f"- **Source:** {src}"]
        if k.get("evidence_quote"):
            lines.append(f"- **Quote:** \"{k['evidence_quote']}\"")
        lines += ["- [ ] OK", ""]
    if rejected:
        lines += ["## Rejected (for reference)", ""]
        lines += [f"- `{r['id']}` {r['question']} — {'; '.join(r['rejected'])}" for r in rejected]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main():
    parser = argparse.ArgumentParser(description="Build a retrieval eval set from test-data PDFs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--files", nargs="+", help="PDF filenames in the PDF dir")
    group.add_argument("--all", action="store_true", help="Use every PDF in the PDF dir")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--out-of-corpus", type=int, default=2)
    parser.add_argument("--near-miss", type=int, default=2)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set in .env")

    files = sorted(args.pdf_dir.glob("*.pdf")) if args.all else [args.pdf_dir / f for f in args.files]
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(8)

    results = await asyncio.gather(*(build_doc(client, args.model, sem, f) for f in files))
    docs = [d for d, _, _ in results]
    kept = [k for _, ks, _ in results for k in ks]
    rejected = [r for _, _, rs in results for r in rs]
    kept += await build_negatives(client, args.model, docs, args.out_of_corpus, args.near_miss)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = [{k: d[k] for k in ("file", "title", "pages")} for d in docs]
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    review = args.out.with_suffix(".review.md")
    write_review(review, kept, rejected, docs)

    print(f"Docs: {len(docs)} | kept: {len(kept)} | rejected: {len(rejected)}")
    print(f"Eval: {args.out}\nReview: {review}")


if __name__ == "__main__":
    asyncio.run(main())
