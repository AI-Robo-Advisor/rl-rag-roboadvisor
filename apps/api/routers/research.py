"""Agentic RAG research endpoint."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from apps.api.observability import log_e2e_event
from apps.api.schemas import ResearchRequest, ResearchResponse
from apps.api.services import build_research_response, stream_research_response

router = APIRouter(tags=["research"])


@router.post("/research", response_model=ResearchResponse)
def research_question(request: ResearchRequest) -> ResearchResponse:
    """Return investment research report data."""
    response = build_research_response(request.question)
    log_e2e_event(
        "/research",
        status=response.status,
        elapsed_ms=response.elapsed_ms,
        timed_out=response.timed_out,
        sources_count=len(response.sources),
        risk_tags_count=len(response.risk_tags),
        risk_signals_count=len(response.risk_signals),
        report_chars=len(response.report),
    )
    return response


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
