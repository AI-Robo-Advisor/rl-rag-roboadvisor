"""Train curve endpoint."""

from typing import Annotated

from fastapi import APIRouter, Query

from apps.api.schemas import BacktestWindow, TrainCurveResponse
from apps.api.services import build_train_curve_response

router = APIRouter(tags=["train_curve"])


@router.get("/train_curve", response_model=TrainCurveResponse)
def get_train_curve(
    wf_window: Annotated[
        BacktestWindow | None,
        Query(description="Walk-forward 윈도우 이름 (w1/w2/w3/final). 미지정 시 가장 최근 run"),
    ] = None,
    smooth_window: Annotated[
        int | None,
        Query(description="이동 평균 스무딩 윈도우 크기 (미지정 시 raw)", ge=1),
    ] = None,
) -> TrainCurveResponse:
    """TensorBoard logs/tensorboard/ppo_*/ 에서 에피소드별 누적 보상을 반환합니다."""
    return build_train_curve_response(wf_window=wf_window, smooth_window=smooth_window)
