import uuid
from dataclasses import dataclass

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.models.document import Document


@dataclass
class BM25Result:
    chunk_id: uuid.UUID
    document_title: str
    page_number: int | None
    content: str
    score: float


async def bm25_search(
    db: AsyncSession,
    tenant_id: uuid.UUID,
    query: str,
    top_k: int,
) -> list[BM25Result]:
    # Parse input query to English search terms
    tsquery = func.plainto_tsquery("english", query)

    # Rank relevance using cover-density
    score = func.ts_rank_cd(Chunk.search_vector, tsquery).label("score")

    # Select chunks matching query, sorted by highest ranking score
    stmt = (
        select(Chunk, Document.title, score)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.tenant_id == tenant_id)
        .where(Chunk.search_vector.op("@@")(tsquery))
        .order_by(desc("score"))
        .limit(top_k)
    )

    rows = await db.execute(stmt)
    results = rows.all()

    return [
        BM25Result(
            chunk_id=chunk.id,
            document_title=doc_title,
            page_number=chunk.page_number,
            content=chunk.content,
            score=float(score_val),
        )
        for chunk, doc_title, score_val in results
    ]
