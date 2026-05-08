"""
금융 뉴스 텍스트에서 리스크 태그를 추출하는 모듈.
규칙 기반(키워드 매핑) 방식으로 리스크 유형을 분류합니다.

RL 관측공간 연동용 3종 태그("규제변경" / "실적쇼크" / "급등락")와
이를 이진 벡터로 변환하는 get_risk_vector()를 포함합니다.
"""
from typing import Dict, List

import numpy as np

# 키워드 → 태그 매핑
# 동일 태그를 가리키는 키워드를 복수로 등록할 수 있습니다.
RISK_KEYWORD_MAP: Dict[str, str] = {
    "금리": "금리_리스크",
    "기준금리": "금리_리스크",
    "인플레이션": "인플레이션_리스크",
    "물가": "인플레이션_리스크",
    "환율": "환율_리스크",
    "달러": "환율_리스크",
    "지정학": "지정학_리스크",
    "전쟁": "지정학_리스크",
    "분쟁": "지정학_리스크",
    "제재": "지정학_리스크",
    "규제": "규제_리스크",
    "법안": "규제_리스크",
    "불확실성": "불확실성_리스크",
    "침체": "경기침체_리스크",
    "리세션": "경기침체_리스크",
    "경기둔화": "경기침체_리스크",
    "부도": "신용_리스크",
    "파산": "신용_리스크",
    "디폴트": "신용_리스크",
    "유동성": "유동성_리스크",
    "변동성": "변동성_리스크",
    "하락": "시장_리스크",
    "폭락": "시장_리스크",
    "급락": "시장_리스크",
}

# 태그별 심각도 분류
_HIGH_TAGS = {"지정학_리스크", "경기침체_리스크", "신용_리스크"}
_MEDIUM_TAGS = {"금리_리스크", "인플레이션_리스크", "유동성_리스크", "시장_리스크"}


def extract_risk_tags(text: str) -> List[str]:
    """
    텍스트에서 리스크 키워드를 탐지하여 태그 리스트를 반환합니다.

    RISK_KEYWORD_MAP을 기반으로 규칙 탐색하며, 중복 태그는 제거됩니다.

    Args:
        text: 분석할 금융 뉴스 텍스트.

    Returns:
        감지된 리스크 태그 리스트 (알파벳/가나다 정렬, 중복 없음).

    Example:
        >>> extract_risk_tags("미국 금리 인상으로 환율 변동성이 커지고 있다.")
        ['금리_리스크', '변동성_리스크', '환율_리스크']
    """
    if not text:
        return []

    found: set = set()
    for keyword, tag in RISK_KEYWORD_MAP.items():
        if keyword in text:
            found.add(tag)

    return sorted(found)


def tag_severity(tag: str) -> str:
    """
    리스크 태그의 심각도 수준을 반환합니다.

    Args:
        tag: 리스크 태그 문자열.

    Returns:
        "HIGH" | "MEDIUM" | "LOW" 중 하나.
    """
    if tag in _HIGH_TAGS:
        return "HIGH"
    if tag in _MEDIUM_TAGS:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────
# RL 관측공간 연동 — 3종 태그
# ─────────────────────────────────────────────

# RL 환경(trading_env.py)과 공유하는 고정 태그 순서 (변경 시 이문정과 협의 필요)
RL_RISK_TAGS: List[str] = ["규제변경", "실적쇼크", "급등락"]

_RL_KEYWORD_MAP: Dict[str, List[str]] = {
    "규제변경": ["규제변경", "규정 변경", "규제 강화", "법 개정", "법개정", "금융 규제", "규제 개편"],
    "실적쇼크": ["실적쇼크", "어닝쇼크", "실적 쇼크", "실적 충격", "실적 부진", "영업손실", "어닝 쇼크"],
    "급등락":   ["급등락", "급등", "급락", "폭등", "폭락", "급변동", "급변"],
}


def extract_rl_risk_tags(text: str) -> List[str]:
    """
    RL 관측공간 연동용 3종 리스크 태그를 추출합니다.

    Args:
        text: 분석할 금융 뉴스 텍스트.

    Returns:
        감지된 태그 리스트. 원소는 RL_RISK_TAGS 내 값만 포함, 정렬됨.

    Example:
        >>> extract_rl_risk_tags("금융당국이 가상자산 규제를 강화했다.")
        ['규제변경']
    """
    if not text:
        return []
    found: set = set()
    for tag, keywords in _RL_KEYWORD_MAP.items():
        if any(kw in text for kw in keywords):
            found.add(tag)
    return sorted(found)


def get_risk_vector(tags: List[str]) -> np.ndarray:
    """
    RL 관측공간 연동용 3차원 이진 벡터를 반환합니다.

    각 차원은 RL_RISK_TAGS 순서에 대응하며, 해당 태그가 있으면 1.0, 없으면 0.0입니다.

    Args:
        tags: extract_rl_risk_tags() 결과 또는 동일 형식의 태그 리스트.

    Returns:
        np.ndarray shape=(3,), dtype=float32.

    Example:
        >>> get_risk_vector(["급등락"])
        array([0., 0., 1.], dtype=float32)
    """
    return np.array(
        [1.0 if t in tags else 0.0 for t in RL_RISK_TAGS],
        dtype=np.float32,
    )


def summarize_risk_profile(tags: List[str]) -> str:
    """
    리스크 태그 리스트를 심각도 기준으로 정리한 요약 문자열을 반환합니다.

    Args:
        tags: 리스크 태그 리스트.

    Returns:
        심각도별 그룹 요약 문자열.

    Example:
        >>> summarize_risk_profile(["금리_리스크", "지정학_리스크"])
        'HIGH: 지정학_리스크 | MEDIUM: 금리_리스크'
    """
    if not tags:
        return "리스크 없음"

    grouped: Dict[str, List[str]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for tag in tags:
        grouped[tag_severity(tag)].append(tag)

    parts = [
        f"{level}: {', '.join(grouped[level])}"
        for level in ("HIGH", "MEDIUM", "LOW")
        if grouped[level]
    ]
    return " | ".join(parts)
