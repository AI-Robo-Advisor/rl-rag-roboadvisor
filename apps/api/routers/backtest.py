"""Backtest endpoint."""

from typing import Annotated

from fastapi import APIRouter, Query

from apps.api.schemas import BacktestResponse, BacktestWindow
from apps.api.services import build_fallback_backtest

router = APIRouter(tags=["backtest"])


@router.get("/backtest", response_model=BacktestResponse)
def get_backtest(
    window: Annotated[BacktestWindow, Query(description="Walk-forward backtest window")] = "final",
) -> BacktestResponse:
    """Return backtest metrics and ANOVA summaries."""
    return build_fallback_backtest(window=window)
