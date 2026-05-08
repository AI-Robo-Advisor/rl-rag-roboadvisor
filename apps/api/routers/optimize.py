"""Portfolio optimization endpoint."""

from fastapi import APIRouter

from apps.api.schemas import OptimizeRequest, OptimizeResponse
from apps.api.services import build_fallback_portfolio

router = APIRouter(tags=["portfolio"])


@router.post("/optimize", response_model=OptimizeResponse)
def optimize_portfolio(request: OptimizeRequest) -> OptimizeResponse:
    """Return normalized portfolio weights."""
    return build_fallback_portfolio(
        tickers=request.tickers,
        risk_profile=request.risk_profile,
        risk_aversion=request.risk_aversion,
    )
