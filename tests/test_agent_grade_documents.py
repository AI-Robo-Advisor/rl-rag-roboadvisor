"""Agent document-grading node tests."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from apps.api.config import settings

settings.OPENAI_API_KEY = settings.OPENAI_API_KEY or "test-key"

from src.agent.nodes import grade_documents_node


def test_grade_documents_logs_retry_refine_latency(monkeypatch) -> None:
    """Retry query refinement should leave a measurable latency log."""
    monkeypatch.setattr(
        "src.agent.nodes._refine_search_query_for_retry",
        lambda state: "보정 쿼리",
    )

    result = grade_documents_node(
        {
            "query": "금리 리스크",
            "plan": "금리와 주식시장 관련 문서를 찾는다",
            "documents": [],
            "distances": [],
            "retry_count": 0,
        }
    )

    assert result["needs_research_retry"] is True
    assert any("retry refine latency=" in message for message in result["messages"])
