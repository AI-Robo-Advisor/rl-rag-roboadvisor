"""
LangGraph StateGraph: Planner → Researcher → grade_documents → (루프) → Analyst.

``grade_documents``에서 검색 품질을 평가하고, 부족하면 researcher로 최대 3회 재진입(Self-Correction)합니다.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import List, Literal

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    AgentState,
    analyst_node,
    grade_documents_node,
    planner_node,
    researcher_node,
)

logger = logging.getLogger(__name__)

# 외부에서 `from src.agent.graph import AgentState` 호환
__all__ = ["AgentState", "build_graph", "graph", "run_graph", "route_after_grade"]


def route_after_grade(state: AgentState) -> Literal["researcher", "analyst"]:
    """
    ``grade_documents`` 직후 분기.

    ``needs_research_retry``가 True이면 researcher로, 아니면 analyst로 갑니다.

    Args:
        state: 갱신된 AgentState.

    Returns:
        다음 노드 키.
    """
    if state.get("needs_research_retry"):
        logger.info("[route] Self-Correction → researcher")
        return "researcher"
    logger.info("[route] → analyst")
    return "analyst"


def build_graph():
    """
    StateGraph를 구성해 컴파일합니다.

    흐름:
        START → planner → researcher → grade_documents
        → (조건) researcher (다시 검색) 또는 analyst → END

    Returns:
        컴파일된 Runnable 그래프.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("analyst", analyst_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"researcher": "researcher", "analyst": "analyst"},
    )
    workflow.add_edge("analyst", END)

    return workflow.compile()


graph = build_graph()


def run_graph(query: str) -> AgentState:
    """
    단일 질의로 전체 워크플로우를 실행합니다.

    Args:
        query: 사용자 금융 질문.

    Returns:
        최종 AgentState(plan, documents, context, risk_tags, response, messages 등).
    """
    initial_state: AgentState = {
        "query": query,
        "messages": [],
        "plan": "",
        "context": "",
        "documents": [],
        "risk_tags": [],
        "distances": [],
        "retry_count": 0,
        "needs_research_retry": False,
        "response": "",
        "sources": [],
        "reasoning_trace": "",
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    """
    데모: ``삼성전자 HBM 실적 전망은?`` 질문으로 노드별 상태 전이를 스트림 출력합니다.

    실행 (프로젝트 루트, PYTHONPATH 설정됨):
        python -m src.agent.graph
    """
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    demo_query = "삼성전자 HBM 실적 전망은?"
    if len(sys.argv) > 1:
        demo_query = " ".join(sys.argv[1:])

    print("질의:", demo_query)
    print("========== LangGraph stream ==========\n")

    g = build_graph()
    base: AgentState = {
        "query": demo_query,
        "messages": [],
        "plan": "",
        "context": "",
        "documents": [],
        "risk_tags": [],
        "distances": [],
        "retry_count": 0,
        "needs_research_retry": False,
        "response": "",
        "sources": [],
        "reasoning_trace": "",
    }

    think_accum: List[str] = []
    for step in g.stream(deepcopy(base)):
        for node_name, payload in step.items():
            print(f"▶ [{node_name}]")
            if not isinstance(payload, dict):
                continue
            p = payload
            if "plan" in p and p["plan"]:
                print(f"  plan: {str(p['plan'])[:300]}…")
            if "documents" in p:
                print(f"  documents: {len(p['documents'])}건")
            if "context" in p and p.get("context"):
                c = str(p["context"])
                print(f"  context: {len(c)}자")
            if "needs_research_retry" in p:
                print(f"  needs_research_retry: {p['needs_research_retry']}")
            if "retry_count" in p:
                print(f"  retry_count: {p['retry_count']}")
            if "response" in p and p.get("response"):
                r = str(p["response"])
                print(f"  response: {r[:400]}{'…' if len(r) > 400 else ''}")
            if "messages" in p and p["messages"]:
                for line in p["messages"]:
                    print(f"  {line}")
                    think_accum.append(line)
            print()

    print("========== THINK 로그 누적 ==========")
    for line in think_accum:
        print(line)
