from fastapi import FastAPI
from apps.api.config import settings

app = FastAPI(title="AI Robo Advisor API")

@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "api_host": settings.API_HOST,
        "api_port": settings.API_PORT,
        "log_level": settings.LOG_LEVEL
    }