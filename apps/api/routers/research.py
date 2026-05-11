"""Agentic RAG research endpoint."""

from fastapi import APIRouter

from apps.api.schemas import ResearchRequest, ResearchResponse
from apps.api.services import build_research_response

router = APIRouter(tags=["research"])


@router.post("/research", response_model=ResearchResponse)
def research_question(request: ResearchRequest) -> ResearchResponse:
    """Return investment research report data."""
    return build_research_response(request.question)
