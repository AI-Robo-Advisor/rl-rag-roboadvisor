"""SHAP explanation endpoint."""

from fastapi import APIRouter

from apps.api.observability import log_e2e_event
from apps.api.schemas import ExplainRequest, ExplainResponse
from apps.api.services import build_explanation_response

router = APIRouter(tags=["explainability"])


@router.post("/explain", response_model=ExplainResponse)
def explain_decision(request: ExplainRequest) -> ExplainResponse:
    """Return feature contributions for a requested date."""
    response = build_explanation_response(request.date, request.top_k)
    feature_reasoning_count = sum(
        len(item.reasoning_context) for item in response.feature_contributions
    )
    log_e2e_event(
        "/explain",
        status=response.status,
        elapsed_ms=response.elapsed_ms,
        timed_out=response.timed_out,
        requested_date=request.date,
        target_date=response.target_date,
        top_k=request.top_k,
        features_count=len(response.feature_contributions),
        reasoning_count=len(response.reasoning_context),
        feature_reasoning_count=feature_reasoning_count,
    )
    return response
