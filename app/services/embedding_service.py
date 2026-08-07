from langfuse.openai import AsyncOpenAI

from app.core.config import settings

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
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
        all_embeddings.extend([r.embedding for r in response.data])

    return all_embeddings
