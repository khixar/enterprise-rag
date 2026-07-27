import asyncio
import json

from langchain_core.runnables import RunnableConfig
from langfuse.decorators import langfuse_context, observe
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.state import AgentState
from app.models.chunk import Chunk
from app.models.document import Document
from app.schemas.query import SourceChunk
from app.services.bm25_service import bm25_search
from app.services.embedding_service import embed_texts, get_client
from app.services.reranker_service import rerank
from app.services.rrf import reciprocal_rank_fusion

_TOP_K = 5
_MIN_SIMILARITY = 0.25


def _db(config: RunnableConfig) -> AsyncSession:
    return config["configurable"]["db"]


@observe(as_type="retriever", name="Retrieve Node")
async def retrieve(state: AgentState, config: RunnableConfig) -> dict:
    db = _db(config)
    question = state["question"]
    tenant_id = state["tenant_id"]
    fetch_k = max(_TOP_K * 3, 15)

    langfuse_context.update_current_observation(
        input={"question": question, "tenant_id": str(tenant_id), "fetch_k": fetch_k}
    )

    query_embedding, bm25_results = await asyncio.gather(
        embed_texts([question]),
        bm25_search(db, tenant_id, question, top_k=_TOP_K * 3),
    )
    query_embedding = query_embedding[0]

    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    rows = await db.execute(
        select(Chunk, Document.title, distance)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.tenant_id == tenant_id)
        .where(Chunk.embedding.isnot(None))
        .where(distance <= (1 - _MIN_SIMILARITY))
        .order_by(distance)
        .limit(fetch_k)
    )
    vector_results = rows.all()
    combined = reciprocal_rank_fusion(vector_results, bm25_results)

    langfuse_context.update_current_observation(
        output={"retrieved_count": len(combined), "retrieval_empty": len(combined) == 0}
    )

    return {
        "retrieved_chunks": combined,
        "retrieval_empty": len(combined) == 0,
    }


_CLASSIFY_PROMPT = """You are a retrieval quality classifier for an HR assistant.

Given a question and retrieved document chunks, decide whether the chunks are relevant
enough to answer the question, or whether this should be escalated (chunks are completely off-topic).

Guidelines:
1. In an HR context, the user might refer to the candidate/employee as 'the user' or 'the candidate'.
2. The phrase 'user experience' or 'user's experience' might refer to the candidate's professional work experience/skills with a technology, not UI/UX design.
3. Do NOT escalate if the primary technology, tool, skill, or topic in the query (e.g. 'Django', 'FastAPI', 'Python', 'AWS') is mentioned or discussed in the chunks. Even if the question is slightly off-topic, as long as it mentions a technology the candidate has worked with, do NOT escalate. Let the answer node handle explaining what is or isn't in the context.
4. Only escalate if the query is about a completely different technology, person, or subject that has zero mention or relevance in the retrieved chunks (e.g. asking about 'React' or 'Cooking recipes' when the chunks are only about a Python backend developer).

Respond with JSON only:
{"should_escalate": true/false, "reason": "<one sentence explaining why, or null if not escalating>"}"""


@observe(as_type="agent", name="Classify Node")
async def classify(state: AgentState, config: RunnableConfig) -> dict:
    question = state["question"]
    chunks = state["retrieved_chunks"]

    langfuse_context.update_current_observation(
        input={
            "question": question,
            "retrieved_count": len(chunks),
            "retrieval_empty": state["retrieval_empty"],
        }
    )

    if state["retrieval_empty"]:
        print("[classify] → escalate (retrieval empty)")
        langfuse_context.update_current_observation(
            output={
                "should_escalate": True,
                "escalation_reason": "No relevant documents were found for this query.",
            }
        )
        return {
            "should_escalate": True,
            "escalation_reason": "No relevant documents were found for this query.",
        }

    context = "\n\n".join(
        f"[{i+1}] ({c.document_title}): {c.content}"
        for i, c in enumerate(chunks[:5])
    )

    client = get_client()
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _CLASSIFY_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nChunks:\n{context}"},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(resp.choices[0].message.content)
    should_escalate = parsed.get("should_escalate", False)
    reason = parsed.get("reason")
    path = "escalate" if should_escalate else "answer"
    print(f"[classify] → {path} | reason: {reason}")

    langfuse_context.update_current_observation(
        output={"should_escalate": should_escalate, "escalation_reason": reason}
    )

    return {
        "should_escalate": should_escalate,
        "escalation_reason": reason,
    }


_ANSWER_PROMPT = """You are an HR assistant. Answer the user's question using only the context provided.
If the answer is not in the context, say you don't have that information.
Be concise and factual."""


@observe(as_type="span", name="Answer Node")
async def answer(state: AgentState, config: RunnableConfig) -> dict:
    question = state["question"]
    chunks = state["retrieved_chunks"]

    langfuse_context.update_current_observation(
        input={"question": question, "candidate_chunks_count": len(chunks)}
    )

    ranked = await rerank(
        query=question,
        documents=[c.content for c in chunks],
        top_n=_TOP_K,
    )

    context_parts = []
    sources: list[SourceChunk] = []
    for r in ranked:
        candidate = chunks[r.index]
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

    client = get_client()
    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _ANSWER_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{'\n\n'.join(context_parts)}\n\nQuestion: {question}",
            },
        ],
        temperature=0.0,
    )

    ans = completion.choices[0].message.content

    langfuse_context.update_current_observation(
        output={"answer": ans, "sources_count": len(sources)}
    )

    return {
        "answer": ans,
        "sources": sources,
    }


@observe(as_type="span", name="Escalate Node")
async def escalate(state: AgentState, config: RunnableConfig) -> dict:
    reason = (
        state.get("escalation_reason")
        or "The question could not be answered from the available documents."
    )

    langfuse_context.update_current_observation(input={"escalation_reason": reason})

    ans = (
        f"I wasn't able to find a reliable answer to your question. "
        f"{reason} Please reach out to your HR team directly."
    )

    langfuse_context.update_current_observation(output={"answer": ans})

    return {
        "answer": ans,
        "sources": [],
    }
