"""Backtest endpoint."""

from fastapi import APIRouter

from apps.api.schemas import BacktestResponse
from apps.api.services import build_fallback_backtest

router = APIRouter(tags=["backtest"])


@router.get("/backtest", response_model=BacktestResponse)
def get_backtest() -> BacktestResponse:
    """Return backtest metrics and ANOVA summaries."""
    return build_fallback_backtest()
