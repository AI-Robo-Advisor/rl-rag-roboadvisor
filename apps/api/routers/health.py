"""Health endpoint."""

from fastapi import APIRouter

from apps.api.config import settings
from apps.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return API readiness and downstream module status."""
    return HealthResponse(
        status="ok",
        api={
            "host": settings.API_HOST,
            "port": settings.API_PORT,
            "log_level": settings.LOG_LEVEL,
        },
        modules={
            "data": "ready",
            "rl": "fallback",
            "rag": "fallback",
            "shap": "fallback",
        },
    )
