from dataclasses import dataclass
import cohere
import asyncio
from langfuse.decorators import langfuse_context, observe

from app.services import usage
from app.core.config import settings

_client: cohere.AsyncClientV2 | None = None
RERANK_MODEL = "rerank-english-v3.0"
_RATE_LIMIT_RETRIES = 12
_RATE_LIMIT_WAIT_S = 6.0

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
        # Trial keys allow 10 calls/min; wait out 429s rather than silently skipping the rerank
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            try:
                response = await get_client().rerank(
                    model=RERANK_MODEL,
                    query=query,
                    documents=documents,
                    top_n=top_n,
                )
                break
            except cohere.TooManyRequestsError:
                if attempt == _RATE_LIMIT_RETRIES:
                    raise
                await asyncio.sleep(_RATE_LIMIT_WAIT_S)
        langfuse_context.update_current_observation(
            output={"results_count": len(response.results)}
        )
        billed = getattr(getattr(response.meta, "billed_units", None), "search_units", None) if response.meta else None
        usage.record(RERANK_MODEL, search_units=int(billed or 1))
        return [
            RankedResult(index=r.index, relevance_score=r.relevance_score)
            for r in response.results
        ]
    except Exception as e:
        # Fallback to original RRF order if Cohere still fails; log it so it can't go unnoticed
        print(f"RERANK_FALLBACK {type(e).__name__}", flush=True)
        top_docs = min(top_n, len(documents))
        return [
            RankedResult(index=idx, relevance_score=1.0 - (idx * 0.1))
            for idx in range(top_docs)
        ]
