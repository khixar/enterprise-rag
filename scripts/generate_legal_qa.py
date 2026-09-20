import asyncio
import json
import random
import sys
from pathlib import Path
from openai import AsyncOpenAI

# Path setup
BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
DOCS_DIR = SCRIPTS_DIR / "legal_docs"
INDEX_FILE = SCRIPTS_DIR / "legal_docs_index.json"
EVAL_FILE = SCRIPTS_DIR / "legal_eval.json"

# Read environment variables / API key
import dotenv
dotenv.load_dotenv(BASE_DIR / ".env")
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PROMPT_TEMPLATE = """You are a legal expert creating ground-truth evaluation questions for a RAG benchmark based on Pakistan Federal Laws.
Below is an excerpt from a legal document:

--- DOCUMENT TITLE: {saved_as} ---
{excerpt}
--- END EXCERPT ---

Generate {count} distinct QA pairs strictly derived from the excerpt above.
Include a mix of:
1. Factual statutory lookups (definitions, authorities, rules).
2. Boundary Value Analysis (BVA) questions if numeric limits/dates/penalties exist (e.g. exact days for appeal, maximum fines, age cutoffs).
3. Procedural / multi-condition requirements.

For each QA pair, provide a JSON object with:
- "question": clear, self-contained legal question.
- "expected_answer": precise, accurate answer based ONLY on the text excerpt.
- "expected_source_document": "{saved_as}"
- "expected_page": null
- "should_retrieve": true

Respond ONLY with a JSON array of objects.
"""

NEGATIVE_PROMPT = """Generate 15 realistic legal questions about Pakistan law that are NOT covered in standard federal acts (e.g. space regulation, 19th-century maritime salvage, cryptocurrency mining licenses, specific provincial municipal bylaws, etc.).

For each negative question, provide:
- "question": realistic legal question.
- "expected_answer": "Not explicitly mentioned in the context."
- "expected_source_document": null
- "expected_page": null
- "should_retrieve": false

Respond ONLY with a JSON array of objects.
"""

async def generate_qa_for_doc(client: AsyncOpenAI, item: dict, count: int = 3) -> list[dict]:
    filepath = DOCS_DIR / item["saved_as"]
    text = filepath.read_text(encoding="utf-8")
    # Take first 4000 characters as context window
    excerpt = text[:4000]
    
    prompt = PROMPT_TEMPLATE.format(
        saved_as=item["saved_as"],
        excerpt=excerpt,
        count=count
    )
    
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        # Handle if wrapped in key like {"questions": [...]} or raw list
        if isinstance(parsed, list):
            return parsed
        for key in ["questions", "qa_pairs", "items", "data"]:
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        return list(parsed.values())[0] if parsed else []
    except Exception as e:
        print(f"Error generating QA for {item['saved_as']}: {e}")
        return []

async def generate_negative_cases(client: AsyncOpenAI) -> list[dict]:
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": NEGATIVE_PROMPT}],
            temperature=0.4,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        for key in ["questions", "negative_cases", "items", "data"]:
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        return list(parsed.values())[0] if parsed else []
    except Exception as e:
        print(f"Error generating negative cases: {e}")
        return []

async def main():
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is not set in environment or .env file.")
        sys.exit(1)
        
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    index_data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    print(f"Loaded index of {len(index_data)} documents.")
    
    # Pick 30 diverse documents (mix of short, medium, long)
    index_data.sort(key=lambda x: x["word_count"])
    selected_docs = index_data[::6][:30] # sampled across spectrum
    
    all_qa = []
    print(f"Generating QA pairs across {len(selected_docs)} sampled legal documents...")
    
    tasks = [generate_qa_for_doc(client, doc, count=3) for doc in selected_docs]
    results = await asyncio.gather(*tasks)
    
    for qa_list in results:
        all_qa.extend(qa_list)
        
    print(f"Generated {len(all_qa)} positive QA pairs.")
    
    print("Generating negative test cases (should_retrieve: false)...")
    negatives = await generate_negative_cases(client)
    all_qa.extend(negatives)
    
    # Shuffle QA list
    random.seed(42)
    random.shuffle(all_qa)
    
    # Write to scripts/legal_eval.json
    EVAL_FILE.write_text(json.dumps(all_qa, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print("\n" + "="*60)
    print(f"QA GENERATION COMPLETE!")
    print(f"Total evaluation cases: {len(all_qa)}")
    print(f"Positive cases (should_retrieve: true) : {sum(1 for c in all_qa if c.get('should_retrieve'))}")
    print(f"Negative cases (should_retrieve: false): {sum(1 for c in all_qa if not c.get('should_retrieve'))}")
    print(f"Output file: {EVAL_FILE}")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
