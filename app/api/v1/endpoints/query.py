from fastapi import APIRouter, Depends
from langfuse.decorators import langfuse_context, observe
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import compiled
from app.db.session import get_db
from app.schemas.query import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
@observe(name="HR Copilot Query API")
async def query(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    langfuse_context.update_current_trace(
        input={"question": request.question, "tenant_id": str(request.tenant_id)},
        session_id=str(request.tenant_id),
    )

    result = await compiled.ainvoke(
        {
            "question": request.question,
            "tenant_id": request.tenant_id,
        },
        config={"configurable": {"db": db}},
    )

    langfuse_context.update_current_trace(output={"answer": result["answer"]})

    return QueryResponse(answer=result["answer"], sources=result["sources"])
