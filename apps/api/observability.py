"""Small structured logs for Docker-based E2E inspection."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("apps.api.e2e")
uvicorn_logger = logging.getLogger("uvicorn.error")


def log_e2e_event(
    endpoint: str,
    *,
    status: str,
    elapsed_ms: float = 0.0,
    timed_out: bool = False,
    **fields: Any,
) -> None:
    """Emit one grep-friendly JSON log line for an API endpoint result."""
    payload: dict[str, Any] = {
        "event": "e2e_endpoint",
        "endpoint": endpoint,
        "status": status,
        "elapsed_ms": round(float(elapsed_ms), 3),
        "timed_out": bool(timed_out),
        **fields,
    }
    message = f"E2E {json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}"
    logger.info(message)
    if uvicorn_logger is not logger:
        uvicorn_logger.info(message)
