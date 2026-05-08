"""FastAPI application entrypoint."""

from fastapi import FastAPI

from apps.api.routers import backtest, explain, health, optimize, research

app = FastAPI(
    title="AI Robo Advisor API",
    description="RL + RAG 기반 로보어드바이저 백엔드",
    version="0.2.0",
)

app.include_router(health.router)
app.include_router(optimize.router)
app.include_router(explain.router)
app.include_router(research.router)
app.include_router(backtest.router)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a minimal root message."""
    return {"message": "FastAPI is running"}
