"""Portfolio optimization endpoint."""

from fastapi import APIRouter

from apps.api.observability import log_e2e_event
from apps.api.schemas import OptimizeRequest, OptimizeResponse
from apps.api.services import build_portfolio_response

router = APIRouter(tags=["portfolio"])


@router.post("/optimize", response_model=OptimizeResponse)
def optimize_portfolio(request: OptimizeRequest) -> OptimizeResponse:
    """Return normalized portfolio weights."""
    response = build_portfolio_response(
        tickers=request.tickers,
        risk_profile=request.risk_profile,
        risk_aversion=request.risk_aversion,
        risk_tags=request.risk_tags,
        risk_signals=request.risk_signals,
    )
    log_e2e_event(
        "/optimize",
        status=response.status,
        elapsed_ms=response.elapsed_ms,
        timed_out=response.timed_out,
        tickers_count=len(response.tickers),
        weights_count=len(response.weights),
        risk_profile=response.risk_profile,
        risk_tags_count=len(request.risk_tags or []),
        risk_signals_count=len(request.risk_signals or []),
    )
    return response
