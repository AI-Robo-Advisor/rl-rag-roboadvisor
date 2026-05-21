"""Agentic RAG research endpoint."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from apps.api.schemas import ResearchRequest, ResearchResponse
from apps.api.services import build_research_response, stream_research_response

router = APIRouter(tags=["research"])


@router.post("/research", response_model=ResearchResponse)
def research_question(request: ResearchRequest) -> ResearchResponse:
    """Return investment research report data."""
    return build_research_response(request.question)


@router.post("/research/stream")
async def stream_research_question(request: ResearchRequest) -> StreamingResponse:
    """Stream research progress as NDJSON for Streamlit st.write_stream()."""
    return StreamingResponse(
        stream_research_response(request.question),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
