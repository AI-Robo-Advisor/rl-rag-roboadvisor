"""SHAP explanation endpoint."""

from fastapi import APIRouter

from apps.api.schemas import ExplainRequest, ExplainResponse
from apps.api.services import build_fallback_explanation

router = APIRouter(tags=["explainability"])


@router.post("/explain", response_model=ExplainResponse)
def explain_decision(request: ExplainRequest) -> ExplainResponse:
    """Return feature contributions for a requested date."""
    return build_fallback_explanation(request.date, request.top_k)
