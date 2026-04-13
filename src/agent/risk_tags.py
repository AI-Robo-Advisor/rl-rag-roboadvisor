"""
금융 뉴스 텍스트에서 리스크 태그를 추출하는 모듈.
규칙 기반(키워드 매핑) 방식으로 리스크 유형을 분류합니다.
"""
from typing import Dict, List

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
