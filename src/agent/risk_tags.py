"""
금융 뉴스 텍스트에서 리스크 태그를 추출하는 모듈.

RL 관측공간 연동용 3종 태그 (macro_rate / equity_market / geopolitical_fx) 와
키워드 밀도 기반 연속 스코어(0.0~1.0) 반환 함수를 포함합니다.

실시간 RAG 파이프라인 전용. 과거 데이터 배치 라벨링은 LLM Batch API 사용
(scripts/build_risk_parquet.py 참고).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

# ─────────────────────────────────────────────
# RL 관측공간 연동 — 3종 태그 (이문정 파트 연동 기준)
# ─────────────────────────────────────────────

RL_RISK_TAGS: List[str] = ["macro_rate", "equity_market", "geopolitical_fx"]

# Exponential decay 기간 (일). build_risk_parquet.py와 동일 값 유지.
DECAY_PERIODS: Dict[str, int] = {
    "macro_rate":      30,   # 금리·매크로 이벤트: 영향 비교적 장기 지속
    "equity_market":   10,   # 시장 변동성: 비교적 빠르게 정상화
    "geopolitical_fx": 60,   # 전쟁·제재: 매우 장기 지속
}

# ─────────────────────────────────────────────
# 키워드 맵 — 방향성(directional) 키워드만 사용
# ─────────────────────────────────────────────

MACRO_KEYWORDS: List[str] = [
    "연준 금리 인상", "기준금리 인상", "금리 인상", "빅스텝", "자이언트 스텝",
    "FOMC 인상", "한국은행 금리 인상",
    "CPI 예상 상회", "인플레이션 장기화", "물가 급등", "인플레 충격",
    "미국 국채금리 급등", "장기금리 상승", "금리 급등",
    "달러 강세", "달러 급등", "DXY 상승",
    "금리 인하 지연", "금리 동결 장기화",
    # English
    "rate hike", "fed rate hike", "fomc hike", "cpi surprise", "inflation surge",
    "dollar surge", "yield surge", "rate freeze",
]

EQUITY_KEYWORDS: List[str] = [
    "증시 급락", "주가 폭락", "코스피 급락", "나스닥 폭락",
    "경기침체 우려", "경기 침체", "리세션", "침체 공포",
    "기업 실적 쇼크", "어닝쇼크", "실적 부진", "어닝 미스",
    "기술주 조정", "성장주 하락", "밸류에이션 부담",
    "VIX 급등", "공포지수 급등",
    "신흥국 자금 유출", "패닉셀", "투매",
    # English
    "market crash", "stock crash", "nasdaq crash", "earnings shock",
    "recession fear", "panic sell", "vix spike",
]

GEO_KEYWORDS: List[str] = [
    "미중 갈등", "미중 무역 분쟁", "무역 전쟁", "관세 부과", "보복 관세",
    "반도체 수출 규제", "수출 통제", "대중 제재",
    "중동 전쟁", "러시아-우크라이나", "지정학 리스크",
    "원유 가격 급등", "유가 급등", "유가 폭등",
    "중국 경기 둔화", "중국 경착륙",
    "원/달러 환율 급등", "원화 약세", "환율 급등",
    "공급망 차질", "공급망 붕괴",
    # English
    "trade war", "tariffs", "sanctions", "geopolitical risk",
    "oil surge", "fx surge", "supply chain disruption",
]

_KEYWORD_MAP: Dict[str, List[str]] = {
    "macro_rate":      MACRO_KEYWORDS,
    "equity_market":   EQUITY_KEYWORDS,
    "geopolitical_fx": GEO_KEYWORDS,
}

# ─────────────────────────────────────────────
# 스코어링 — 키워드 밀도 (실시간 RAG용)
# ─────────────────────────────────────────────

_MAX_HITS = 3  # 키워드 3개 이상 히트 시 1.0


def keyword_density(text: str, keywords: List[str]) -> float:
    """키워드 밀도 기반 리스크 강도를 반환합니다.

    Args:
        text: 분석할 텍스트 (title + summary 결합 권장).
        keywords: 해당 태그의 키워드 목록.

    Returns:
        0.0 ~ 1.0 사이의 강도 값.
    """
    text_cf = text.casefold()
    hits = sum(1 for kw in keywords if kw.casefold() in text_cf)
    return round(min(hits / _MAX_HITS, 1.0), 2)


def score_risk_vector(text: str) -> Tuple[float, float, float]:
    """텍스트에서 3종 리스크 스코어를 반환합니다 (실시간 RAG용).

    Args:
        text: 분석할 텍스트.

    Returns:
        (macro_rate, equity_market, geopolitical_fx) 스코어 튜플.
    """
    if not text:
        return (0.0, 0.0, 0.0)
    return (
        keyword_density(text, MACRO_KEYWORDS),
        keyword_density(text, EQUITY_KEYWORDS),
        keyword_density(text, GEO_KEYWORDS),
    )


def get_risk_vector(text: str) -> np.ndarray:
    """RL 관측공간 연동용 shape=(3,) float32 벡터를 반환합니다.

    Args:
        text: 분석할 텍스트.

    Returns:
        np.ndarray shape=(3,), dtype=float32.
        순서: [macro_rate, equity_market, geopolitical_fx].
    """
    return np.array(score_risk_vector(text), dtype=np.float32)


# ─────────────────────────────────────────────
# Exponential Decay — 실시간 단일 이벤트 보정용
# ─────────────────────────────────────────────

def apply_decay(severity: float, days_elapsed: int, tag: str) -> float:
    """단일 이벤트의 decay 적용 강도를 반환합니다.

    과거 parquet의 raw_severity를 실시간으로 보정할 때 사용.
    배치 decay 계산은 scripts/build_risk_parquet.py 참고.

    Args:
        severity: 원본 강도 (0.0~1.0).
        days_elapsed: 이벤트 발생 이후 경과 일수.
        tag: 태그 이름 ("macro_rate" / "equity_market" / "geopolitical_fx").

    Returns:
        decay 적용된 강도 (0.0~1.0).
    """
    period = DECAY_PERIODS.get(tag, 30)
    return round(severity * math.exp(-days_elapsed / period), 4)


# ─────────────────────────────────────────────
# 일반 RAG 디스플레이용 (기존 호환 유지)
# ─────────────────────────────────────────────

def extract_risk_tags(text: str) -> List[str]:
    """텍스트에서 활성화된 리스크 태그 목록을 반환합니다 (RAG 디스플레이용).

    Args:
        text: 분석할 텍스트.

    Returns:
        활성화된 태그 리스트. 스코어 > 0인 태그만 포함.
    """
    if not text:
        return []
    macro, equity, geo = score_risk_vector(text)
    tags: List[str] = []
    if macro > 0:
        tags.append("macro_rate_risk")
    if equity > 0:
        tags.append("equity_market_risk")
    if geo > 0:
        tags.append("geopolitical_fx_risk")
    return tags


def extract_rl_risk_tags(text: str) -> List[str]:
    """RL 관측공간 연동용 태그 리스트를 반환합니다.

    Args:
        text: 분석할 텍스트.

    Returns:
        활성화된 RL_RISK_TAGS 원소 리스트.
    """
    if not text:
        return []
    scores = score_risk_vector(text)
    return [tag for tag, score in zip(RL_RISK_TAGS, scores) if score > 0]


def summarize_risk_profile(tags: List[str]) -> str:
    """리스크 태그 리스트를 요약 문자열로 반환합니다.

    Args:
        tags: 리스크 태그 리스트.

    Returns:
        요약 문자열.
    """
    if not tags:
        return "리스크 없음"
    return " | ".join(tags)
