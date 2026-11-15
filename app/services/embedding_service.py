from langfuse.openai import AsyncOpenAI

from app.core.config import settings
from app.services import usage

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        # The SDK default (600s timeout, 2 retries) let one stalled request block an upload for 30 min
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=60.0)
    return _client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, batching into groups of 128."""
    client = get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), 128):
        batch = texts[i : i + 128]
        response = await client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=batch,
        )
        usage.record(settings.EMBEDDING_MODEL, input_tokens=response.usage.prompt_tokens)
        all_embeddings.extend([r.embedding for r in response.data])

    return all_embeddings
