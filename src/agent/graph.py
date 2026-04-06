"""
LangGraph 기반 RAG 워크플로우 정의 모듈.
Planner → Researcher → Analyst 순서로 실행되는 StateGraph를 구성합니다.
"""
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from src.agent.nodes import analyst_node, planner_node, researcher_node


class AgentState(TypedDict):
    """
    LangGraph 워크플로우 전체 상태 스키마.

    Attributes:
        query:     사용자의 원본 질의.
        plan:      Planner 노드가 생성한 조사 계획.
        documents: Researcher 노드가 검색한 문서 리스트.
                   각 항목은 {"content": str, "metadata": dict} 형태.
        risk_tags: 검색 문서에서 추출한 리스크 태그 리스트.
        response:  Analyst 노드의 최종 분석 응답.
    """

    query: str
    plan: str
    documents: List[Dict[str, Any]]
    risk_tags: List[str]
    response: str


def build_graph():
    """
    Planner → Researcher → Analyst 순서의 LangGraph를 빌드합니다.

    Returns:
        컴파일된 LangGraph 실행 체인 (CompiledGraph).

    Graph Flow::

        START → planner → researcher → analyst → END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", END)

    return workflow.compile()


# 모듈 레벨 싱글톤 그래프
graph = build_graph()


def run_graph(query: str) -> AgentState:
    """
    사용자 쿼리를 받아 전체 RAG 워크플로우를 실행합니다.

    Args:
        query: 사용자의 금융 관련 질의.

    Returns:
        최종 AgentState.
        plan, documents, risk_tags, response 필드가 채워진 상태.

    Example:
        >>> result = run_graph("삼성전자 투자 리스크를 분석해줘")
        >>> print(result["response"])
    """
    initial_state: AgentState = {
        "query": query,
        "plan": "",
        "documents": [],
        "risk_tags": [],
        "response": "",
    }
    return graph.invoke(initial_state)
