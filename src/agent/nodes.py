"""
LangGraph 워크플로우 노드 구현 모듈.

Planner, Researcher, grade_documents(자기교정 판별), Analyst 노드를 정의합니다.
각 노드는 대시보드 연동을 위해 ``THINK`` 형식의 사고 로그를 남깁니다.
"""
from __future__ import annotations

import logging
import operator
import os
from typing import Annotated, Any, Dict, List, NotRequired, TypedDict

import chromadb.errors
import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from apps.api.config import settings
from src.agent.risk_tags import extract_risk_tags
from src.agent.vectorstore import collection_document_count, query_documents

logger = logging.getLogger(__name__)

# 모든 노드가 공유하는 LLM 인스턴스
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=settings.OPENAI_API_KEY,
)


class AgentState(TypedDict):
    """
    LangGraph StateGraph에서 공유하는 에이전트 상태 스키마.

    Attributes:
        query: 사용자 원본 질의.
        messages: 노드별 사고·진행 로그(누적). 대시보드 노출용 한 줄 문자열.
        plan: Planner가 만든 조사 계획.
        context: Researcher가 만든 RAG 컨텍스트 문자열.
        documents: 검색 문서 ``{"content", "metadata"}`` 리스트.
        risk_tags: ``risk_tags`` 모듈 추출 태그.
        distances: Chroma 거리 목록(문서 순). 없으면 빈 리스트.
        retry_count: Self-Correction 재검색 횟수.
        needs_research_retry: ``True``면 다음 노드가 researcher.
        response: Analyst 최종 답변.
        search_query: 재검색 시 우선 사용할 보정 쿼리(없으면 미설정).
    """

    query: str
    messages: Annotated[List[str], operator.add]
    plan: str
    context: str
    documents: List[Dict[str, Any]]
    risk_tags: List[str]
    distances: List[float]
    retry_count: int
    needs_research_retry: bool
    response: str
    search_query: NotRequired[str]


def _think_log(node: str, detail: str) -> str:
    """대시보드·콘솔용 통일 로그 한 줄을 만듭니다."""
    line = f"[THINK][{node}] {detail}"
    logger.info("%s", line)
    return line


def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    사용자 질문을 분석하여 뉴스·통계 검색에 쓸 조사 계획(플랜)을 세웁니다.

    OpenAI 챗 모델로 1~3문장 분량의 검색 지침을 생성하고,
    현재 사고 과정을 ``messages``에 남깁니다.

    Args:
        state: ``query``가 채워진 AgentState.

    Returns:
        ``plan`` 및 누적될 ``messages`` 조각.

    Raises:
        없음 (LLM 오류는 LangChain 예외로 상위 전파).
    """
    query = state["query"]
    detail = f"질의 분석 시작: {query[:120]}{'…' if len(query) > 120 else ''}"
    msg = _think_log("planner", detail)

    sys = SystemMessage(
        content=(
            "당신은 금융 투자 리서치 플래너입니다. "
            "사용자 질의를 바탕으로 벡터 DB(뉴스·경제통계) 검색에 쓸 "
            "간결한 조사 계획을 1~3문장 한국어로 작성하세요."
        )
    )
    hum = HumanMessage(content=f"사용자 질의:\n{query}")
    try:
        plan = llm.invoke([sys, hum]).content
    except openai.OpenAIError as e:
        logger.error("planner: LLM 호출 실패 (%s). 원 질의를 플랜으로 대체합니다.", e)
        plan = query
    after = _think_log("planner", f"플랜 확정(요약): {plan[:200]}{'…' if len(plan) > 200 else ''}")
    return {"plan": plan, "messages": [msg, after]}


def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    ``vectorstore.query_documents``로 ChromaDB에서 관련 문서를 검색합니다.

    ``search_query``가 있으면(재검색 루프) 이를 우선 사용하고, 없으면 ``query + plan``을
    합친 문자열로 검색합니다. 결과가 없을 때 예외를 포획해 빈 결과로 안전하게 진행합니다.

    Args:
        state: ``query``, ``plan`` 필수. ``search_query``, ``retry_count`` 선택.

    Returns:
        ``documents``, ``context``, ``risk_tags``, ``distances``, ``messages``.
    """
    query = state["query"]
    plan = state.get("plan") or ""
    refined = (state.get("search_query") or "").strip()
    retry = int(state.get("retry_count") or 0)

    if refined:
        search_text = refined
        open_msg = _think_log(
            "researcher",
            f"재검색({retry}회차) 쿼리 사용: {search_text[:160]}{'…' if len(search_text) > 160 else ''}",
        )
    else:
        search_text = f"{query} {plan}".strip()
        open_msg = _think_log(
            "researcher",
            f"초기 검색: {search_text[:160]}{'…' if len(search_text) > 160 else ''}",
        )

    documents: List[Dict[str, Any]] = []
    distances: List[float] = []
    persist = settings.CHROMA_PERSIST_DIR
    abs_persist = os.path.abspath(persist)
    total_cnt = collection_document_count(persist)

    try:
        results = query_documents(
            query_texts=[search_text],
            n_results=5,
            persist_dir=persist,
        )
        if results.get("documents") and results["documents"][0]:
            metas = results["metadatas"][0] if results.get("metadatas") else []
            dists = results["distances"][0] if results.get("distances") else []
            for i, content in enumerate(results["documents"][0]):
                meta = metas[i] if i < len(metas) else {}
                documents.append({"content": content, "metadata": meta})
            if dists and len(dists) == len(documents):
                distances = list(dists)
        if not documents:
            mid = _think_log(
                "researcher",
                f"Chroma hit=0건 — persist={abs_persist}, "
                f"컬렉션 총 {total_cnt}건. "
                f"비어 있으면 프로젝트 루트에서 collector_smoke_test 또는 news_collector 적재 실행.",
            )
            logger.warning(
                "RAG 결과 없음: path=%s count=%s query_preview=%s",
                abs_persist,
                total_cnt,
                search_text[:100],
            )
        else:
            mid = _think_log(
                "researcher",
                f"Chroma hit={len(documents)}건 (top-k=5, DB총 {total_cnt}건, {abs_persist})",
            )
    except (chromadb.errors.ChromaError, ValueError, OSError) as e:
        # ChromaDB 관련 예외(컬렉션 오류, 잘못된 파라미터, DB 파일 I/O)만 포획.
        # 그 외 예상치 못한 예외는 상위로 전파하여 파이프라인이 명시적으로 실패하도록 함.
        logger.exception("researcher: vectorstore 조회 실패")
        mid = _think_log(
            "researcher",
            f"검색 실패(예외). 빈 결과로 진행: {type(e).__name__}: {e}",
        )

    all_text = " ".join(d["content"] for d in documents)
    risk_tags = extract_risk_tags(all_text) if all_text else []

    context_parts: List[str] = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.get("metadata") or {}
        context_parts.append(
            f"[문서 {i}] {meta.get('title', '제목 없음')}\n"
            f"날짜: {meta.get('date', '')}\n"
            f"본문: {doc['content']}\n"
            f"출처: {meta.get('url', '')}"
        )
    context = "\n\n".join(context_parts) if context_parts else "관련 문서 없음"

    tail = _think_log(
        "researcher",
        f"컨텍스트 길이={len(context)}자, 리스크태그={risk_tags or '없음'}",
    )
    return {
        "documents": documents,
        "context": context,
        "risk_tags": risk_tags,
        "distances": distances,
        "messages": [open_msg, mid, tail],
    }


def _refine_search_query_for_retry(state: AgentState) -> str:
    """
    Self-Correction용으로 검색 문장을 한 줄로 다시 씁니다.

    Args:
        state: query, plan 및 이전 search_query를 알고 있는 상태.

    Returns:
        새 검색 문자열.
    """
    query = state["query"]
    plan = state.get("plan") or ""
    prev = (state.get("search_query") or "").strip() or f"{query} {plan}".strip()
    sys = SystemMessage(
        content=(
            "당신은 검색어 보정기입니다. 뉴스·통계 벡터 검색에 맞게 "
            "핵심 키워드를 살린 한국어 문장 **한 줄**만 출력하세요. 따옴표·부가 설명 없음."
        )
    )
    hum = HumanMessage(
        content=(
            f"원 질의: {query}\n"
            f"플랜: {plan}\n"
            f"이전 검색문: {prev}\n"
            f"위 검색은 결과가 부족했습니다. 더 넓거나 다른 표현의 검색문 한 줄:"
        )
    )
    try:
        return llm.invoke([sys, hum]).content.strip()
    except openai.OpenAIError as e:
        logger.error("refine_query: LLM 호출 실패 (%s). 이전 쿼리를 유지합니다.", e)
        return prev


def grade_documents_node(state: AgentState) -> Dict[str, Any]:
    """
    검색 결과의 양·유사도를 평가해 Self-Correction 여부를 결정합니다.

    - 문서 0건이거나 2건 미만이면 부족으로 간주합니다.
    - 코사인 거리가 있다면, **가장 좋은(최소) 거리**가 임계값을 넘으면 부족으로 간주합니다.
    - ``retry_count``가 3 이상이면 더 이상 researcher로 돌리지 않고 통과시킵니다.

    재검색이 필요하면 ``search_query``를 LLM으로 보정하고 ``documents``·``context``를 비웁니다.

    Args:
        state: Researcher 직후 상태.

    Returns:
        ``needs_research_retry``, ``retry_count``, ``search_query`` 등 갱신 필드.
    """
    docs = state.get("documents") or []
    dists = list(state.get("distances") or [])
    retry = int(state.get("retry_count") or 0)

    msgs: List[str] = []
    msgs.append(_think_log("grade_documents", f"평가 시작: 문서 {len(docs)}건, retry_count={retry}"))

    insufficient = False
    if not docs:
        insufficient = True
        msgs.append(_think_log("grade_documents", "판정: 결과 없음 → 불충분"))
    elif len(docs) < 2:
        insufficient = True
        msgs.append(_think_log("grade_documents", "판정: 문서 수 < 2 → 불충분"))
    elif dists and len(dists) == len(docs):
        best = min(float(x) for x in dists)
        # ChromaDB 코사인 거리: 0(완전 일치) ~ 1(완전 다름).
        # 0.75 초과 = 상위 결과조차 관련성이 낮다고 판단. 실험적으로 설정한 임계값 (조정 가능).
        if best > 0.75:
            insufficient = True
            msgs.append(
                _think_log(
                    "grade_documents",
                    f"판정: 최소거리 {best:.4f} > 0.75 → 관련성 낮음",
                )
            )

    if insufficient and retry < 3:
        new_q = _refine_search_query_for_retry(state)
        msgs.append(_think_log("grade_documents", f"재검색 결정 ({retry + 1}/3). 신규 쿼리: {new_q[:120]}…"))
        return {
            "messages": msgs,
            "needs_research_retry": True,
            "retry_count": retry + 1,
            "search_query": new_q,
            "documents": [],
            "context": "",
            "risk_tags": [],
            "distances": [],
        }

    if insufficient:
        msgs.append(
            _think_log(
                "grade_documents",
                "불충분이나 retry 상한(3) 도달 — 현재 결과로 analyst 진행",
            )
        )
    else:
        msgs.append(_think_log("grade_documents", "판정: 충분 — analyst 진행"))

    return {
        "messages": msgs,
        "needs_research_retry": False,
    }


def analyst_node(state: AgentState) -> Dict[str, Any]:
    """
    ``context``와 리스크 태그를 근거로 최종 투자 관점 의견을 작성합니다.

    Args:
        state: ``query``, ``context``, ``risk_tags`` 권장.

    Returns:
        ``response`` 및 로그 ``messages``.
    """
    query = state["query"]
    context = state.get("context") or "관련 문서 없음"
    risk_tags: List[str] = state.get("risk_tags") or []

    msg = _think_log("analyst", "최종 리포트 생성 착수")
    risk_summary = ", ".join(risk_tags) if risk_tags else "없음"

    sys = SystemMessage(
        content=(
            "당신은 전문 금융 투자 애널리스트입니다. "
            "제공 컨텍스트가 빈약하면 그 한계를 명시하고, "
            "있을 경우 시장 동향·기회·리스크를 한국어로 간결히 정리하세요."
        )
    )
    hum = HumanMessage(
        content=(
            f"[질의]\n{query}\n\n"
            f"[근거 컨텍스트]\n{context}\n\n"
            f"[리스크 태그]\n{risk_summary}"
        )
    )
    try:
        response = llm.invoke([sys, hum]).content
    except openai.OpenAIError as e:
        logger.error("analyst: LLM 호출 실패 (%s).", e)
        response = f"[분석 오류] LLM 호출에 실패했습니다: {type(e).__name__}"
    tail = _think_log(
        "analyst",
        f"응답 길이={len(response)}자 (미리보기: {response[:80]}…)",
    )
    return {
        "response": response,
        "messages": [msg, tail],
    }
