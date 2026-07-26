import asyncio
import uuid

from langfuse.decorators import langfuse_context, observe
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.models.document import Document
from app.schemas.query import QueryResponse, SourceChunk
from app.services.bm25_service import bm25_search
from app.services.embedding_service import embed_texts, get_client
from app.services.reranker_service import rerank
from app.services.rrf import reciprocal_rank_fusion

SYSTEM_PROMPT = """You are an HR assistant. Answer the user's question using only the context provided.
If the answer is not in the context, say you don't have that information.
Be concise and factual."""


@observe(name="Legacy Run Query Pipeline")
async def run_query(
    db: AsyncSession,
    tenant_id: uuid.UUID,
    question: str,
    top_k: int = 5,
    min_similarity: float = 0.25,
) -> QueryResponse:
    langfuse_context.update_current_observation(
        input={
            "question": question,
            "tenant_id": str(tenant_id),
            "top_k": top_k,
            "min_similarity": min_similarity,
        }
    )
    query_embedding, bm25_results = await asyncio.gather(
        embed_texts([question]),
        bm25_search(db, tenant_id, question, top_k=top_k * 3),
    )
    query_embedding = query_embedding[0]

    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    fetch_k = max(top_k * 3, 15)

    rows = await db.execute(
        select(Chunk, Document.title, distance)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.tenant_id == tenant_id)
        .where(Chunk.embedding.isnot(None))
        .where(distance <= (1 - min_similarity))
        .order_by(distance)
        .limit(fetch_k)
    )
    vector_results = rows.all()

    if not vector_results and not bm25_results:
        return QueryResponse(
            answer="No relevant documents found for this tenant.", sources=[]
        )

    combined = reciprocal_rank_fusion(vector_results, bm25_results)

    ranked = await rerank(
        query=question,
        documents=[c.content for c in combined],
        top_n=top_k,
    )

    context_parts = []
    sources: list[SourceChunk] = []

    for r in ranked:
        candidate = combined[r.index]
        location = (
            f"p.{candidate.page_number}" if candidate.page_number else "unknown page"
        )
        context_parts.append(
            f"[{len(sources)+1}] ({candidate.document_title}, {location})\n{candidate.content}"
        )
        sources.append(
            SourceChunk(
                document_title=candidate.document_title,
                page_number=candidate.page_number,
                content=candidate.content,
                rrf_score=round(candidate.rrf_score, 6),
                relevance_score=round(r.relevance_score, 6),
            )
        )

    if not context_parts:
        return QueryResponse(
            answer="No relevant documents found for this query.", sources=[]
        )

    context = "\n\n".join(context_parts)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    client = get_client()
    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )

    ans = completion.choices[0].message.content
    langfuse_context.update_current_observation(
        output={"answer": ans, "sources_count": len(sources)}
    )

    return QueryResponse(
        answer=ans,
        sources=sources,
    )
