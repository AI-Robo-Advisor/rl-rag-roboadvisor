"""
RAG 품질 평가 스크립트 (강유영 담당).

docs/rag_eval_questions.json의 12개 질문으로 ChromaDB 검색 성능을 측정합니다.

지표:
  - 검색 히트율: top-1 cosine distance < 0.4 비율
  - 태그 정확도: 검색 문서의 risk_label + 키워드 추출이 expected tag를 포함하는 비율

실행:
  PYTHONPATH=. python scripts/rag_eval.py
  PYTHONPATH=. python scripts/rag_eval.py --n_results 5  # top-k 변경
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

logging.basicConfig(level=logging.WARNING)  # chromadb 로그 억제

EVAL_PATH = Path("docs/rag_eval_questions.json")
HIT_THRESHOLD = 0.4  # cosine distance < 임계값 → 히트


def _found_tags_from_results(results: Dict[str, Any]) -> Set[str]:
    """검색 결과에서 risk 태그를 추출합니다."""
    from src.agent.risk_tags import extract_risk_tags

    tags: Set[str] = set()
    docs  = results["documents"][0]
    metas = results["metadatas"][0]

    for meta in metas:
        for tag in meta.get("risk_label", "").split(","):
            tag = tag.strip()
            if tag:
                tags.add(tag)

    for doc in docs:
        for tag in extract_risk_tags(doc):
            tags.add(tag)

    return tags


def run_eval(n_results: int = 3) -> Dict[str, Any]:
    from src.agent.vectorstore import query_documents

    questions: List[Dict[str, Any]] = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    hit_count = 0
    tag_count = 0
    rows: List[Dict[str, Any]] = []

    for q in questions:
        results = query_documents(query_texts=[q["question"]], n_results=n_results)
        dists   = results["distances"][0]
        docs    = results["documents"][0]
        metas   = results["metadatas"][0]

        top_dist = dists[0] if dists else 1.0
        is_hit   = top_dist < HIT_THRESHOLD

        found  = _found_tags_from_results(results)
        expect = set(q["expect_rl_tags"])
        tag_ok = bool(expect & found)

        if is_hit:
            hit_count += 1
        if tag_ok:
            tag_count += 1

        rows.append({
            "id":         q["id"],
            "question":   q["question"],
            "expect":     list(expect),
            "found_tags": list(found),
            "top_dist":   round(top_dist, 4),
            "hit":        is_hit,
            "tag_ok":     tag_ok,
            "top_doc":    docs[0][:80] if docs else "",
            "top_date":   metas[0].get("date", "?")[:10] if metas else "?",
        })

    total = len(questions)
    return {
        "n_questions":    total,
        "n_results":      n_results,
        "hit_threshold":  HIT_THRESHOLD,
        "hit_rate":       round(hit_count / total, 4),
        "tag_accuracy":   round(tag_count / total, 4),
        "details":        rows,
    }


def print_report(report: Dict[str, Any]) -> None:
    print(f"\n=== RAG 품질 평가 (top-{report['n_results']}, dist<{report['hit_threshold']}) ===")
    print(f"{'ID':<5} {'HIT':>3} {'TAG':>3} {'DIST':>6}  질문 / 1위 문서")
    print("-" * 80)

    for r in report["details"]:
        h = "✅" if r["hit"]    else "❌"
        t = "✅" if r["tag_ok"] else "❌"
        print(f"{r['id']:<5} {h}  {t}  {r['top_dist']:>6.3f}  {r['question'][:35]}")
        print(f"      기대:{r['expect']} / 검색:{r['found_tags']}")
        print(f"      [{r['top_date']}] {r['top_doc']}")

    print("-" * 80)
    print(f"검색 히트율 (dist<{report['hit_threshold']}): "
          f"{round(report['hit_rate']*report['n_questions'])}/{report['n_questions']} "
          f"= {report['hit_rate']:.1%}")
    print(f"태그 정확도:                          "
          f"{round(report['tag_accuracy']*report['n_questions'])}/{report['n_questions']} "
          f"= {report['tag_accuracy']:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 품질 평가")
    parser.add_argument("--n_results", type=int, default=3,
                        help="ChromaDB 검색 상위 k개 (기본값: 3)")
    parser.add_argument("--json", action="store_true",
                        help="결과를 JSON으로 출력")
    args = parser.parse_args()

    report = run_eval(n_results=args.n_results)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
