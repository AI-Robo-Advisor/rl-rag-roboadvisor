"""
RAG 품질 평가 배치 스크립트.

docs/rag_eval_questions.json의 질문을 LangGraph 그래프로 실행하고
hit_count, min_distance, retry_count, has_sources, report_len, rl_risk_tags 등의
지표를 기록해 docs/rag_eval_results.md에 저장합니다.

실행:
  python scripts/run_rag_eval.py
  python scripts/run_rag_eval.py --dry-run   # Chroma 검색만, LLM 호출 없음
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path("docs/rag_eval_questions.json")
RESULTS_PATH   = Path("docs/rag_eval_results.md")


def run_graph_query(question: str) -> dict[str, Any]:
    """LangGraph 그래프 실행 후 상태 지표 반환."""
    from src.agent.graph import build_graph

    graph = build_graph()
    state = graph.invoke({"query": question, "messages": []})

    documents  = state.get("documents", [])
    distances  = state.get("distances", [])
    rl_tags    = state.get("rl_risk_tags", [])
    response   = state.get("response", "")
    retry      = state.get("retry_count", 0)

    hit_count    = len(documents)
    min_distance = round(min(distances), 4) if distances else None
    sources_list = [
        d.get("metadata", {}).get("url", "")
        for d in documents
        if d.get("metadata", {}).get("url", "")
    ]
    has_sources = len(sources_list) > 0
    report_len  = len(response)

    return {
        "hit_count":    hit_count,
        "min_distance": min_distance,
        "retry_count":  retry,
        "has_sources":  has_sources,
        "sources_count": len(sources_list),
        "rl_risk_tags": rl_tags,
        "report_len":   report_len,
        "response_preview": response[:120].replace("\n", " "),
    }


def run_dry_query(question: str) -> dict[str, Any]:
    """ChromaDB 검색만 실행 (LLM 없음)."""
    from src.agent.vectorstore import get_vectorstore

    vs = get_vectorstore()
    results = vs.similarity_search_with_relevance_scores(question, k=5)

    documents = [{"content": d.page_content, "metadata": d.metadata} for d, _ in results]
    distances = [1.0 - score for _, score in results]  # relevance → distance

    hit_count    = len(documents)
    min_distance = round(min(distances), 4) if distances else None
    sources_list = [
        d["metadata"].get("url", "")
        for d in documents
        if d["metadata"].get("url", "")
    ]

    return {
        "hit_count":    hit_count,
        "min_distance": min_distance,
        "retry_count":  0,
        "has_sources":  len(sources_list) > 0,
        "sources_count": len(sources_list),
        "rl_risk_tags": [],
        "report_len":   0,
        "response_preview": "(dry-run: LLM 미실행)",
    }


def fmt_tags(tags: list[str]) -> str:
    return ", ".join(f"`{t}`" for t in tags) if tags else "—"


def write_results(questions: list[dict], results: list[dict]) -> None:
    lines = [
        "# RAG 평가 결과",
        "",
        "| id | 질문 | hit | min_dist | retry | sources | rl_tags | report_len |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    hit2_count = 0
    for q, r in zip(questions, results):
        qtext = q["question"][:30] + "…" if len(q["question"]) > 30 else q["question"]
        hit = r["hit_count"]
        dist = r["min_distance"] if r["min_distance"] is not None else "—"
        retry = r["retry_count"]
        src = r["sources_count"]
        tags = fmt_tags(r["rl_risk_tags"])
        rlen = r["report_len"]
        lines.append(
            f"| {q['id']} | {qtext} | {hit} | {dist} | {retry} | {src} | {tags} | {rlen} |"
        )
        if hit >= 2:
            hit2_count += 1

    total = len(questions)
    lines += [
        "",
        "## 요약",
        "",
        f"- **총 질문 수**: {total}개",
        f"- **hit ≥ 2**: {hit2_count}/{total} ({hit2_count/total:.0%})",
        "",
        "## 수동 검수 항목 (상위 5개)",
        "",
        "| id | 관련성 (Y/N) | 출처 (Y/N) | 태그 (Y/N) |",
        "| --- | --- | --- | --- |",
    ]
    for q in questions[:5]:
        lines.append(f"| {q['id']} | — | — | — |")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n결과 저장: {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="LLM 없이 Chroma 검색만 실행")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    results: list[dict] = []

    run_fn = run_dry_query if args.dry_run else run_graph_query

    print(f"{'[DRY RUN] ' if args.dry_run else ''}질문 {len(questions)}개 평가 시작\n")
    print(f"{'id':<5} {'hit':>4} {'min_dist':>9} {'retry':>6} {'src':>4} {'rl_tags':<35} {'report_len':>10}")
    print("-" * 80)

    for q in questions:
        try:
            r = run_fn(q["question"])
        except Exception as e:
            logger.error("오류 [%s]: %s", q["id"], e)
            r = {
                "hit_count": 0, "min_distance": None, "retry_count": 0,
                "has_sources": False, "sources_count": 0, "rl_risk_tags": [],
                "report_len": 0, "response_preview": f"ERROR: {e}",
            }
        results.append(r)

        dist_str = f"{r['min_distance']:.4f}" if r["min_distance"] is not None else "  N/A"
        tags_str = ",".join(r["rl_risk_tags"]) if r["rl_risk_tags"] else "—"
        print(
            f"{q['id']:<5} {r['hit_count']:>4} {dist_str:>9} {r['retry_count']:>6} "
            f"{r['sources_count']:>4} {tags_str:<35} {r['report_len']:>10}"
        )

    write_results(questions, results)

    hit2 = sum(1 for r in results if r["hit_count"] >= 2)
    print(f"\nhit ≥ 2: {hit2}/{len(questions)} ({hit2/len(questions):.0%})")


if __name__ == "__main__":
    main()
