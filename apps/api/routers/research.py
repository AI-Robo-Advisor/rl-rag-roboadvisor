"""Agentic RAG research endpoint."""

from fastapi import APIRouter

from apps.api.schemas import ResearchRequest, ResearchResponse
from apps.api.services import build_fallback_research

router = APIRouter(tags=["research"])


@router.post("/research", response_model=ResearchResponse)
def research_question(request: ResearchRequest) -> ResearchResponse:
    """Return investment research report data."""
    return build_fallback_research(request.question)
