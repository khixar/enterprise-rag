import json
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.models.document import Document
from app.services import usage
from app.services.embedding_service import embed_texts, get_client
from app.services.parsing import extract_text, extract_text_by_page, split_sections, split_text

_TITLE_PROMPT = """Below is the opening text of a document (possibly OCR, may contain errors).
Return its official title. For legislation use the short title, e.g.
"The Domestic Violence (Prevention and Protection) Act, 2026". Correct obvious OCR errors.

Respond with JSON only: {"title": "..."}"""


async def _document_title(opening_text: str, fallback: str) -> str:
    """Ask the LLM for the document's real title; filenames are often meaningless hashes."""
    try:
        resp = await get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _TITLE_PROMPT},
                {"role": "user", "content": opening_text[:3000]},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        usage.record("gpt-4o-mini", resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return json.loads(resp.choices[0].message.content).get("title") or fallback
    except Exception:
        return fallback


async def ingest_document(
    db: AsyncSession,
    tenant_id: uuid.UUID,
    filename: str,
    content: bytes,
    mime_type: str,
) -> tuple[Document, int]:
    if mime_type == "application/pdf":
        pages = extract_text_by_page(content)
        opening_text = " ".join(text for _, text in pages[:2])
        sections = split_sections(pages)
        chunk_texts = [s.text for s in sections]
        labels = [s.label for s in sections]
        page_numbers: list[int | None] = [s.page_number for s in sections]
    else:
        full_text = extract_text(content, mime_type)
        opening_text = full_text
        chunk_texts = split_text(full_text)
        labels = [""] * len(chunk_texts)
        page_numbers = [None] * len(chunk_texts)

    # Prefix every chunk with the document title (and section, when known) so embeddings,
    # FTS and the answer prompt all know where a chunk belongs
    doc_title = await _document_title(opening_text, fallback=filename)
    chunk_texts = [
        f"[{doc_title} — {label[:150]}]\n{c}" if label else f"[{doc_title}]\n{c}"
        for c, label in zip(chunk_texts, labels)
    ]

    embeddings = await embed_texts(chunk_texts)

    document = Document(tenant_id=tenant_id, title=filename, mime_type=mime_type)
    db.add(document)
    await db.flush()

    chunks = [
        Chunk(
            tenant_id=tenant_id,
            document_id=document.id,
            content=chunk_texts[i],
            chunk_index=i,
            page_number=page_numbers[i],
            embedding=embeddings[i],
            extra={"doc_title": doc_title, "section": labels[i] or None},
        )
        for i in range(len(chunk_texts))
    ]
    db.add_all(chunks)
    await db.commit()
    await db.refresh(document)

    return document, len(chunks)
