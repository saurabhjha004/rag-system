from fastapi import APIRouter, Request
from pydantic import BaseModel


# ---------------------------------------------------------
# Router
# ---------------------------------------------------------

router = APIRouter()


# ---------------------------------------------------------
# Request Model
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


# ---------------------------------------------------------
# Response Models
# ---------------------------------------------------------

class Source(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


# ---------------------------------------------------------
# Query Endpoint
# ---------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse
)
def query(
    request: QueryRequest,
    http_request: Request
):
    rag_service = http_request.app.state.rag_service

    result = rag_service.ask(request.query)

    return result