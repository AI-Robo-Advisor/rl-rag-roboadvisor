"""Backtest endpoint."""

from typing import Annotated

from fastapi import APIRouter, Query

from apps.api.observability import log_e2e_event
from apps.api.schemas import BacktestResponse, BacktestWindow
from apps.api.services import build_backtest_response

router = APIRouter(tags=["backtest"])


@router.get("/backtest", response_model=BacktestResponse)
def get_backtest(
    window: Annotated[BacktestWindow, Query(description="Walk-forward backtest window")] = "final",
) -> BacktestResponse:
    """Return backtest metrics and ANOVA summaries."""
    response = build_backtest_response(window=window)
    log_e2e_event(
        "/backtest",
        status=response.status,
        elapsed_ms=response.elapsed_ms,
        timed_out=response.timed_out,
        window=window,
        metrics_count=len(response.metrics),
        anova_count=len(response.anova),
        safeguard_active=response.safeguard.active,
    )
    return response
