from dataclasses import dataclass
import cohere
import asyncio
from langfuse.decorators import langfuse_context, observe
from app.core.config import settings

_client: cohere.AsyncClientV2 | None = None
RERANK_MODEL = "rerank-english-v3.0"

def get_client() -> cohere.AsyncClientV2:
    global _client
    if _client is None:
        _client = cohere.AsyncClientV2(api_key=settings.COHERE_API_KEY)
    return _client

@dataclass
class RankedResult:
    index: int
    relevance_score: float

@observe(as_type="span", name="Cohere Rerank")
async def rerank(query: str, documents: list[str], top_n: int) -> list[RankedResult]:
    if not documents:
        return []
    langfuse_context.update_current_observation(
        input={"query": query, "documents_count": len(documents), "top_n": top_n}
    )
    try:
        response = await get_client().rerank(
            model=RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=top_n,
        )
        langfuse_context.update_current_observation(
            output={"results_count": len(response.results)}
        )
        return [
            RankedResult(index=r.index, relevance_score=r.relevance_score)
            for r in response.results
        ]
    except Exception as e:
        # Fallback to original RRF order if Cohere rate-limits or fails
        top_docs = min(top_n, len(documents))
        return [
            RankedResult(index=idx, relevance_score=1.0 - (idx * 0.1))
            for idx in range(top_docs)
        ]
