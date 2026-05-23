"""build_risk_parquet sanity 테스트.

알려진 고위험 날짜에서 올바른 dominant axis가 나오는지 검증합니다.
risk_vectors_daily.parquet가 없으면 테스트를 스킵합니다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PARQUET_PATH = Path("data/processed/risk_vectors_daily.parquet")


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    if not PARQUET_PATH.exists():
        pytest.skip(f"parquet 없음: {PARQUET_PATH} — scripts/build_risk_parquet.py 먼저 실행")
    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


# ── 날짜 범위 ──────────────────────────────────────────────────────

def test_date_range_starts_2018(daily: pd.DataFrame) -> None:
    assert daily.index.min().year == 2018


def test_date_range_ends_2025(daily: pd.DataFrame) -> None:
    assert daily.index.max().year == 2025


def test_row_count(daily: pd.DataFrame) -> None:
    # 2018-01-01 ~ 2025-12-31 = 2922일
    assert len(daily) >= 2920


# ── sanity: 알려진 고위험 날짜 dominant axis ────────────────────────
# 기대값을 set으로 받아 multi-axis 충격일(동률 1.0) 케이스도 명시한다.
# (design_baseline.md §5-pre 결정 C)

@pytest.mark.parametrize("date_str,expected_top", [
    ("2022-06-15", {"macro"}),                  # FOMC 자이언트 스텝 — macro 단독 우세
    ("2020-03-16", {"macro", "equity", "geo"}), # 코로나 패닉 — 다축 동시 충격(전 축 1.0 포화)
    ("2022-02-24", {"geo"}),                    # 러-우 전쟁 개전
    ("2022-10-07", {"geo"}),                    # 미국 반도체 수출 규제
])
def test_dominant_axis(daily: pd.DataFrame, date_str: str, expected_top: set[str]) -> None:
    ts = pd.Timestamp(date_str)
    assert ts in daily.index, f"{date_str} 날짜가 parquet에 없음"
    row = daily.loc[ts]
    vals = {
        "macro":  float(row["risk_macro"]),
        "equity": float(row["risk_equity"]),
        "geo":    float(row["risk_geo"]),
    }
    top_value = max(vals.values())
    top_keys = {k for k, v in vals.items() if v == top_value}
    assert top_keys == expected_top, (
        f"{date_str}: top_keys={top_keys} (기대={expected_top}), vals={vals}"
    )


# ── 값 범위 ───────────────────────────────────────────────────────

def test_values_in_range(daily: pd.DataFrame) -> None:
    for col in ["risk_macro", "risk_equity", "risk_geo"]:
        assert (daily[col] >= 0.0).all(), f"{col}에 음수 있음"
        assert (daily[col] <= 1.0).all(), f"{col}에 1.0 초과 있음"


# ── max pooling 효과: 과포화 방지 ─────────────────────────────────

def test_no_saturation(daily: pd.DataFrame) -> None:
    # max pooling 후 각 축이 전체 날짜의 10% 이상 1.0으로 포화되면 안 됨
    threshold = 0.10
    n = len(daily)
    for col in ["risk_macro", "risk_equity", "risk_geo"]:
        saturated_ratio = (daily[col] >= 0.99).sum() / n
        assert saturated_ratio < threshold, (
            f"{col} 포화율 {saturated_ratio:.1%} ≥ {threshold:.0%} — max pooling 확인 필요"
        )
