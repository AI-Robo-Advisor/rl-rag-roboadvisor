"""
LangGraph 워크플로우 노드 구현 모듈.
Planner, Researcher, Analyst 세 가지 노드를 정의합니다.
"""
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from apps.api.config import settings
from src.agent.risk_tags import extract_risk_tags
from src.agent.vectorstore import query_documents

# 모든 노드가 공유하는 LLM 인스턴스
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=settings.OPENAI_API_KEY,
)


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    사용자 질의를 분석하여 조사 계획을 수립하는 노드.

    LLM을 통해 어떤 키워드·주제로 뉴스를 검색할지
    1~3문장 분량의 계획을 생성합니다.

    Args:
        state: AgentState 딕셔너리. ``query`` 필드 필요.

    Returns:
        ``plan`` 필드가 업데이트된 딕셔너리.
    """
    query = state["query"]

    messages = [
        SystemMessage(
            content=(
                "당신은 금융 투자 리서치 플래너입니다. "
                "사용자의 질의를 분석하여 어떤 정보를 검색해야 하는지 "
                "간결한 조사 계획을 1~3문장으로 작성하세요."
            )
        ),
        HumanMessage(content=f"사용자 질의: {query}"),
    ]

    response = llm.invoke(messages)
    return {"plan": response.content}


def researcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    벡터스토어에서 관련 뉴스 문서를 검색하는 노드.

    Planner가 생성한 plan과 원본 query를 결합하여
    ChromaDB에서 코사인 유사도 기반 Top-5 문서를 가져옵니다.
    검색 결과로부터 리스크 태그도 함께 추출합니다.

    Args:
        state: AgentState 딕셔너리. ``query``, ``plan`` 필드 필요.

    Returns:
        ``documents``, ``risk_tags`` 필드가 업데이트된 딕셔너리.
    """
    query = state["query"]
    plan = state["plan"]

    # query + plan 결합으로 검색 정확도 향상
    search_text = f"{query} {plan}"

    results = query_documents(query_texts=[search_text], n_results=5)

    documents: List[Dict[str, Any]] = []
    if results.get("documents") and results["documents"][0]:
        for content, meta in zip(results["documents"][0], results["metadatas"][0]):
            documents.append({"content": content, "metadata": meta})

    # 검색된 전체 텍스트에서 리스크 태그 추출
    all_text = " ".join(d["content"] for d in documents)
    risk_tags = extract_risk_tags(all_text)

    return {"documents": documents, "risk_tags": risk_tags}


def analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    검색된 문서를 종합하여 투자 분석 응답을 생성하는 노드.

    뉴스 문서와 리스크 태그를 LLM 컨텍스트로 구성하고,
    투자자를 위한 금융 분석 리포트를 작성합니다.

    Args:
        state: AgentState 딕셔너리.
               ``query``, ``documents``, ``risk_tags`` 필드 필요.

    Returns:
        ``response`` 필드가 업데이트된 딕셔너리.
    """
    query = state["query"]
    documents: List[Dict[str, Any]] = state["documents"]
    risk_tags: List[str] = state["risk_tags"]

    # 뉴스 컨텍스트 구성
    context_parts = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.get("metadata", {})
        context_parts.append(
            f"[뉴스 {i}] {meta.get('title', '제목 없음')}\n"
            f"날짜: {meta.get('date', '날짜 없음')}\n"
            f"내용: {doc['content']}\n"
            f"출처: {meta.get('url', '')}"
        )
    context = "\n\n".join(context_parts) if context_parts else "관련 뉴스 없음"
    risk_summary = ", ".join(risk_tags) if risk_tags else "없음"

    messages = [
        SystemMessage(
            content=(
                "당신은 전문 금융 투자 애널리스트입니다. "
                "제공된 뉴스 기사를 바탕으로 투자자를 위한 분석 리포트를 작성하세요. "
                "주요 시장 동향, 투자 기회, 리스크 요인을 포함하고 한국어로 작성하세요."
            )
        ),
        HumanMessage(
            content=(
                f"[질의]\n{query}\n\n"
                f"[관련 뉴스]\n{context}\n\n"
                f"[감지된 리스크 태그]\n{risk_summary}"
            )
        ),
    ]

    response = llm.invoke(messages)
    return {"response": response.content}
