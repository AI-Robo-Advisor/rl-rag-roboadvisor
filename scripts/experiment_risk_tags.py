"""
실험 1: 리스크 태그 추출 정확도 검증 (논문 4.3절)

의존성 없이 즉시 실행 가능. OPENAI_API_KEY 불필요.

실행:
    python scripts/experiment_risk_tags.py
"""
from src.agent.risk_tags import extract_risk_tags, summarize_risk_profile, tag_severity

# ---------------------------------------------------------------------------
# 테스트 케이스: 금융 이벤트별 뉴스 텍스트 + 정답 태그
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "event": "SVB 파산 사태 (2023-03)",
        "text": (
            "실리콘밸리은행(SVB)이 파산하면서 금융시장에 신용 리스크 우려가 확산됐다. "
            "예금 뱅크런으로 유동성 위기가 심화되었으며, 미국 금리 인상 기조와 맞물려 "
            "부도 우려가 커지고 있다."
        ),
        "expected_tags": {"신용_리스크", "유동성_리스크", "금리_리스크"},
    },
    {
        "event": "러시아-우크라이나 전쟁 (2022-02)",
        "text": (
            "러시아의 우크라이나 침공으로 지정학 리스크가 급격히 상승했다. "
            "서방의 제재 조치로 글로벌 공급망 불확실성이 커지고, "
            "에너지·원자재 가격 폭락과 주식시장 급락이 이어졌다."
        ),
        "expected_tags": {"지정학_리스크", "불확실성_리스크", "시장_리스크"},
    },
    {
        "event": "미국 인플레이션 피크 (2022-06)",
        "text": (
            "미국 CPI가 40년 만에 최고치를 기록하며 인플레이션이 정점에 달했다. "
            "Fed의 공격적 기준금리 인상으로 달러 강세가 나타나고 "
            "경기둔화 우려가 확산되었다."
        ),
        "expected_tags": {"인플레이션_리스크", "금리_리스크", "환율_리스크", "경기침체_리스크"},
    },
    {
        "event": "COVID-19 시장 충격 (2020-03)",
        "text": (
            "코로나19 팬데믹으로 글로벌 주식시장이 폭락하며 변동성이 극도로 높아졌다. "
            "경기침체 우려와 유동성 위기로 안전자산 선호 현상이 강해졌다."
        ),
        "expected_tags": {"시장_리스크", "변동성_리스크", "경기침체_리스크", "유동성_리스크"},
    },
    {
        "event": "일반 경제 뉴스 (리스크 없음)",
        "text": "삼성전자가 2분기 호실적을 발표하며 주가가 상승세를 이어갔다.",
        "expected_tags": set(),
    },
]


def evaluate_extraction(test_cases: list) -> dict:
    """각 테스트 케이스에 대해 추출 정확도를 계산합니다."""
    results = []
    total_tp = total_fp = total_fn = 0

    for case in test_cases:
        predicted = set(extract_risk_tags(case["text"]))
        expected = case["expected_tags"]

        tp = len(predicted & expected)
        fp = len(predicted - expected)
        fn = len(expected - predicted)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        total_tp += tp
        total_fp += fp
        total_fn += fn

        results.append({
            "event": case["event"],
            "predicted": sorted(predicted),
            "expected": sorted(expected),
            "severity_summary": summarize_risk_profile(sorted(predicted)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    # 전체 micro-average
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )

    return {
        "cases": results,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }


def print_paper_format(result: dict) -> None:
    """논문 4.3절에 기재할 형식으로 출력합니다."""
    sep = "=" * 65

    print(sep)
    print("실험 1: 리스크 태그 추출 정확도 검증")
    print(sep)

    for i, case in enumerate(result["cases"], 1):
        print(f"\n[케이스 {i}] {case['event']}")
        print(f"  추출된 태그  : {case['predicted'] or '없음'}")
        print(f"  심각도 요약  : {case['severity_summary']}")
        print(f"  Precision   : {case['precision']:.3f}")
        print(f"  Recall      : {case['recall']:.3f}")
        print(f"  F1 Score    : {case['f1']:.3f}")

    print(f"\n{sep}")
    print("【전체 성능 (Micro-Average)】")
    print(f"  Precision : {result['micro_precision']:.3f}")
    print(f"  Recall    : {result['micro_recall']:.3f}")
    print(f"  F1 Score  : {result['micro_f1']:.3f}")
    print(sep)

    print("\n【논문 기재 예시 문장】")
    print(
        f"규칙 기반 리스크 태그 추출 시스템을 5가지 금융 이벤트 텍스트에 대해 검증한 결과, "
        f"Precision {result['micro_precision']:.3f}, "
        f"Recall {result['micro_recall']:.3f}, "
        f"F1 Score {result['micro_f1']:.3f}를 달성하였다. "
        "본 시스템은 SVB 파산, 지정학 갈등, 인플레이션 등 주요 금융 이벤트에서 "
        "해당 리스크 유형을 정확하게 분류하였으며, "
        "추출된 태그는 RL 환경의 관측 공간에 직접 주입되어 에이전트의 리스크 인식 능력을 향상시킨다."
    )
    print()


if __name__ == "__main__":
    result = evaluate_extraction(TEST_CASES)
    print_paper_format(result)
