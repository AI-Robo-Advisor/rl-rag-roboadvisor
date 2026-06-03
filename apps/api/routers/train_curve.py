"""Train curve endpoint."""

from typing import Annotated

from fastapi import APIRouter, Query

from apps.api.schemas import TrainCurveResponse
from apps.api.services import build_train_curve_response

router = APIRouter(tags=["train_curve"])


@router.get("/train_curve", response_model=TrainCurveResponse)
def get_train_curve(
    window: Annotated[
        int | None,
        Query(description="이동 평균 스무딩 윈도우 크기 (미지정 시 raw)", ge=1),
    ] = None,
) -> TrainCurveResponse:
    """TensorBoard logs/tensorboard/ppo_*/ 에서 에피소드별 누적 보상을 반환합니다."""
    return build_train_curve_response(window=window)
