import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.document import Document
from app.schemas.document import DocumentRead
from app.services import tenant_service
from app.services.document_service import ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_MIME_TYPES = {"application/pdf", "text/plain"}


class DocumentInfo(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    mime_type: str | None

    model_config = {"from_attributes": True}


@router.get("/", response_model=list[DocumentInfo])
async def list_documents(
    tenant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    tenant = await tenant_service.get_tenant(db, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    stmt = (
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .order_by(Document.created_at.desc())
    )
    result = await db.execute(stmt)
    documents = result.scalars().all()
    return [
        DocumentInfo(
            id=doc.id,
            title=doc.title,
            created_at=doc.created_at,
            mime_type=doc.mime_type,
        )
        for doc in documents
    ]


@router.post("/upload", response_model=DocumentRead, status_code=201)
async def upload_document(
    tenant_id: uuid.UUID,
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{file.content_type}'. Use PDF or plain text.")

    tenant = await tenant_service.get_tenant(db, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    document, chunk_count = await ingest_document(
        db=db,
        tenant_id=tenant_id,
        filename=file.filename or "untitled",
        content=content,
        mime_type=file.content_type,
    )

    return DocumentRead(
        id=document.id,
        tenant_id=document.tenant_id,
        title=document.title,
        source_uri=document.source_uri,
        mime_type=document.mime_type,
        created_at=document.created_at,
        chunk_count=chunk_count,
    )
