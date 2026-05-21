"""
LLM 배치 라벨링 스크립트 (강유영 담당).

data/raw/{gdelt,fred,ecos,manual_seed}/ 의 parquet을 통합하고
OpenAI Batch API(gpt-4o-mini)로 리스크 태그를 라벨링합니다.

입력:
  label_method == 'rule_based' | 'manual' → 이미 라벨 있음, 스킵
  label_method == 'llm'                  → Batch API 라벨링 대상

출력:
  data/processed/unified_events.parquet
  컬럼: (공통 스키마) + macro_rate_risk | equity_market_risk |
        geopolitical_fx_risk | primary_tag | reasoning | confidence

실행:
  python scripts/label_events_llm.py                 # 전체 파이프라인
  python scripts/label_events_llm.py --submit-only   # Batch 제출만
  python scripts/label_events_llm.py --collect-only  # 결과 수집만
  python scripts/label_events_llm.py --validate      # 품질 검증만

사전 준비:
  .env에 OPENAI_API_KEY 설정
  pip install openai pandas pyarrow python-dotenv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

RAW_PATHS: List[Path] = [
    Path("data/raw/gdelt/gdelt_events_2018_2025.parquet"),
    Path("data/raw/fred/fred_events_2018_2025.parquet"),
    Path("data/raw/ecos/ecos_events_2018_2025.parquet"),
    Path("data/raw/manual_seed/manual_seed_events.parquet"),
]

OUTPUT_PATH      = Path("data/processed/unified_events.parquet")
BATCH_INPUT_PATH = Path("data/processed/llm_batch_input.jsonl")
BATCH_META_PATH  = Path("data/processed/llm_batch_meta.json")

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS  = 200

# 유효 리스크 강도 값
VALID_RISK_VALUES = {0.0, 0.33, 0.66, 1.0}
VALID_TAGS = {"macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk", "none"}

# ─────────────────────────────────────────────
# 프롬프트
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 금융 시장 리스크를 분류하는 전문가입니다.
주어진 뉴스 기사를 분석하여 3개의 리스크 카테고리에 대해 강도를 평가합니다.

# 리스크 카테고리 정의

## 1. macro_rate_risk (거시 금리 리스크)
- 의미: 금리, 인플레이션, 중앙은행 정책, 채권금리, 달러 강세 관련 리스크
- 영향 자산: 미국 장기채(TLT), 한국 국고채(114260), 리츠(VNQ), 성장주(QQQ), 금(GLD)
- 활성화 키워드: FOMC, Fed, 기준금리, CPI, PPI, 인플레이션, 국채금리, 한국은행, 달러 강세, DXY

## 2. equity_market_risk (주식시장 리스크)
- 의미: 주식시장 전반 하락 위험, 경기침체, 실적 둔화, 변동성 확대
- 영향 자산: SPY, QQQ, IWM, EFA, EEM, KODEX 200
- 활성화 키워드: 증시 급락, 경기침체, 리세션, 어닝 쇼크, 기술주 조정, VIX 급등, 패닉셀

## 3. geopolitical_fx_risk (지정학·환율 리스크)
- 의미: 전쟁, 미중 갈등, 수출 규제, 공급망 충격, 환율 리스크, 원자재 가격 충격
- 영향 자산: 금(GLD), 신흥국(EEM), 선진국(EFA), 한국(KODEX 200), 반도체 노출(QQQ)
- 활성화 키워드: 미중 갈등, 무역 전쟁, 관세, 러우 전쟁, 중동 전쟁, 반도체 수출 규제, 공급망, 원/달러 환율

# 강도 분류 기준 (4단계)

- 0.0  (해당 없음): 해당 카테고리와 무관한 뉴스
- 0.33 (낮음): 단순 언급, 예상 범위 내 관측, 일반적 시황
- 0.66 (중간): 명확한 이벤트 발생, 시장 반응 예상되는 발표
- 1.0  (높음): 충격 이벤트, 즉각적 시장 반응 발생, 정책 변경 또는 위기

# 분류 규칙

1. 언어 무관: 영문/국문 뉴스 모두 동일한 기준으로 분류한다.
2. 다중 활성화 허용: 하나의 뉴스가 여러 카테고리에 영향을 줄 수 있다.
   예: "Fed 금리 인상으로 신흥국 자금 유출" → macro_rate_risk=1.0, geopolitical_fx_risk=0.66
3. primary_tag: 3개 중 가장 강한 영향을 받는 카테고리를 선택한다. 모두 0.0이면 "none".
4. 추측 금지: 명시적으로 언급되지 않은 영향은 0.0으로 평가한다.
5. 단순 시황 보도: "오늘 코스피 0.5% 상승"처럼 큰 이벤트 없는 일상 시황은 모두 0.0으로 평가한다.
6. reasoning: 50자 이내로 핵심 키워드만 인용하며 분류 근거를 작성한다.
7. confidence: 카테고리/강도 판단에 대한 본인의 확신도를 평가한다.
   - 1.0: 명확한 이벤트, 의심의 여지 없음
   - 0.66: 합리적 판단이나 다른 해석 가능
   - 0.33: 모호한 표현이나 정보 부족"""

# few-shot 예시를 messages 형식으로 변환
FEW_SHOT_PAIRS: List[Tuple[str, str]] = [
    (
        "제목: Fed, 기준금리 0.75%p 인상 단행…22년 만의 자이언트 스텝\n"
        "게시일: 2022-06-15\n"
        "본문 요약: 미 연준이 22년 만에 0.75%포인트 금리 인상을 단행했다. 파월 의장은 인플레이션 억제를 위해 추가 인상 가능성도 시사했다.\n"
        "출처: fred",
        '{"macro_rate_risk": 1.0, "equity_market_risk": 0.66, "geopolitical_fx_risk": 0.33, '
        '"primary_tag": "macro_rate_risk", "reasoning": "자이언트 스텝 금리 인상, 22년 만의 충격 이벤트", "confidence": 1.0}',
    ),
    (
        "제목: 러시아, 우크라이나 침공 개시…전면전 돌입\n"
        "게시일: 2022-02-24\n"
        "본문 요약: 러시아군이 우크라이나 전역에서 군사작전을 시작했다. 국제 유가는 배럴당 100달러를 돌파했고 안전자산 선호 심리가 확산되고 있다.\n"
        "출처: gdelt",
        '{"macro_rate_risk": 0.33, "equity_market_risk": 0.66, "geopolitical_fx_risk": 1.0, '
        '"primary_tag": "geopolitical_fx_risk", "reasoning": "전쟁 발발, 유가 100달러 돌파, 안전자산 선호", "confidence": 1.0}',
    ),
    (
        "제목: SVB 파산에 미 증시 폭락…나스닥 -2.05%, KBW은행지수 -7.7%\n"
        "게시일: 2023-03-10\n"
        "본문 요약: 실리콘밸리은행(SVB) 파산 우려로 미국 은행주가 폭락했다. 시장 전체로 패닉 매도가 확산되며 VIX는 28을 돌파했다.\n"
        "출처: gdelt",
        '{"macro_rate_risk": 0.66, "equity_market_risk": 1.0, "geopolitical_fx_risk": 0.0, '
        '"primary_tag": "equity_market_risk", "reasoning": "SVB 파산 패닉, 은행주 폭락, VIX 28 돌파", "confidence": 1.0}',
    ),
    (
        "제목: 9월 미국 CPI 3.7%…시장 예상치 부합\n"
        "게시일: 2023-10-12\n"
        "본문 요약: 미국 9월 소비자물가지수가 전년 동월 대비 3.7% 상승해 시장 예상치에 부합했다. Fed의 추가 금리 인상 가능성은 낮아진 것으로 분석된다.\n"
        "출처: fred",
        '{"macro_rate_risk": 0.66, "equity_market_risk": 0.33, "geopolitical_fx_risk": 0.0, '
        '"primary_tag": "macro_rate_risk", "reasoning": "CPI 발표, 예상치 부합으로 충격 제한적", "confidence": 1.0}',
    ),
    (
        "제목: 파월 의장 \"인플레이션 둔화 추세 지속 중\"\n"
        "게시일: 2023-08-25\n"
        "본문 요약: 잭슨홀 미팅에서 파월 의장은 인플레이션이 둔화되고 있으나 목표치까지 갈 길이 멀다고 언급했다.\n"
        "출처: gdelt",
        '{"macro_rate_risk": 0.33, "equity_market_risk": 0.0, "geopolitical_fx_risk": 0.0, '
        '"primary_tag": "macro_rate_risk", "reasoning": "Fed 의장 발언, 단순 톤 코멘트", "confidence": 0.66}',
    ),
    (
        "제목: 美, 대중 반도체 수출 추가 규제…삼성·SK 우려 확산\n"
        "게시일: 2022-10-07\n"
        "본문 요약: 미국이 대중국 반도체 장비 수출 규제를 대폭 강화했다. 삼성전자, SK하이닉스 등 한국 반도체 업계도 영향권에 들었으며 코스피는 1.8% 하락했다.\n"
        "출처: gdelt",
        '{"macro_rate_risk": 0.0, "equity_market_risk": 0.66, "geopolitical_fx_risk": 1.0, '
        '"primary_tag": "geopolitical_fx_risk", "reasoning": "반도체 수출 규제, 한국 반도체 충격, 코스피 하락", "confidence": 1.0}',
    ),
    (
        "제목: 코스피, 외국인 매수에 0.4% 상승 마감\n"
        "게시일: 2023-07-15\n"
        "본문 요약: 코스피가 외국인 순매수에 힘입어 0.4% 상승 마감했다. 거래량은 평년 수준을 유지했다.\n"
        "출처: manual_seed",
        '{"macro_rate_risk": 0.0, "equity_market_risk": 0.0, "geopolitical_fx_risk": 0.0, '
        '"primary_tag": "none", "reasoning": "일상 시황, 특이 이벤트 없음", "confidence": 1.0}',
    ),
    (
        "제목: 엔비디아 어닝 서프라이즈, AI 수요 폭증 확인\n"
        "게시일: 2024-02-21\n"
        "본문 요약: 엔비디아가 시장 예상을 크게 상회하는 실적을 발표했다. 데이터센터 매출이 전년 대비 409% 증가했다.\n"
        "출처: gdelt",
        '{"macro_rate_risk": 0.0, "equity_market_risk": 0.0, "geopolitical_fx_risk": 0.0, '
        '"primary_tag": "none", "reasoning": "긍정적 실적, 리스크 신호 아님", "confidence": 1.0}',
    ),
]

# structured output schema (OpenAI response_format)
RISK_TAG_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "risk_tag_label",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "macro_rate_risk":     {"type": "number", "enum": [0.0, 0.33, 0.66, 1.0]},
                "equity_market_risk":  {"type": "number", "enum": [0.0, 0.33, 0.66, 1.0]},
                "geopolitical_fx_risk":{"type": "number", "enum": [0.0, 0.33, 0.66, 1.0]},
                "primary_tag":         {"type": "string", "enum": ["macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk", "none"]},
                "reasoning":           {"type": "string"},
                "confidence":          {"type": "number", "enum": [0.33, 0.66, 1.0]},
            },
            "required": [
                "macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk",
                "primary_tag", "reasoning", "confidence",
            ],
            "additionalProperties": False,
        },
    },
}


def _build_few_shot_messages() -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    for user_text, assistant_text in FEW_SHOT_PAIRS:
        msgs.append({"role": "user",      "content": _user_prompt(user_text)})
        msgs.append({"role": "assistant", "content": assistant_text})
    return msgs


def _user_prompt(text: str) -> str:
    return (
        "다음 뉴스를 분석하여 리스크 태그를 분류해주세요.\n\n"
        "# 뉴스 정보\n"
        f"{text}\n\n"
        "위 시스템 지침과 예시에 따라 JSON 형식으로 출력하세요."
    )


def _news_text(row: pd.Series) -> str:
    return (
        f"제목: {row.get('title', '')}\n"
        f"게시일: {str(row.get('date', ''))[:10]}\n"
        f"본문 요약: {row.get('summary', '')}\n"
        f"출처: {row.get('source', '')}"
    )


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────

def load_all_events() -> pd.DataFrame:
    """모든 raw parquet을 통합합니다. 누락 파일은 스킵."""
    frames: List[pd.DataFrame] = []
    for path in RAW_PATHS:
        if not path.exists():
            logger.info("스킵 (파일 없음): %s", path)
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
        logger.info("로드: %s (%d건)", path.name, len(df))

    if not frames:
        logger.error("로드된 파일이 없습니다.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["event_id"], keep="first")
    combined = combined.sort_values("date").reset_index(drop=True)
    logger.info("통합 이벤트: %d건 (중복 제거 후)", len(combined))
    return combined


def split_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """rule_based/manual은 그대로, llm 대상만 분리합니다."""
    is_llm = df["label_method"] == "llm"
    # 이미 라벨이 채워진 llm 행은 재라벨링 스킵
    already_labeled = is_llm & df.get("macro_rate_risk", pd.Series(dtype=float)).notna()
    need_label = is_llm & ~already_labeled
    done = df[~need_label].copy()
    todo = df[need_label].copy()
    logger.info("이미 라벨 있음: %d건 / LLM 라벨링 필요: %d건", len(done), len(todo))
    return done, todo


# ─────────────────────────────────────────────
# Batch API
# ─────────────────────────────────────────────

def build_batch_jsonl(todo: pd.DataFrame) -> None:
    """Batch API 입력 JSONL 파일을 생성합니다."""
    few_shot_msgs = _build_few_shot_messages()
    BATCH_INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(BATCH_INPUT_PATH, "w", encoding="utf-8") as f:
        for idx, (_, row) in enumerate(todo.iterrows()):
            custom_id = f"news-{idx}-{row['event_id']}"
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *few_shot_msgs,
                        {"role": "user", "content": _user_prompt(_news_text(row))},
                    ],
                    "response_format": RISK_TAG_SCHEMA,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    logger.info("JSONL 생성: %s (%d건)", BATCH_INPUT_PATH, len(todo))


def submit_batch(client: Any) -> str:
    """Batch API에 JSONL을 제출하고 batch_id를 반환합니다."""
    with open(BATCH_INPUT_PATH, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    logger.info("파일 업로드: %s", uploaded.id)

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "risk_tag_labeling_v1"},
    )
    logger.info("Batch 제출 완료: id=%s, status=%s", batch.id, batch.status)

    # 메타 저장 (나중에 --collect-only 때 사용)
    meta = {
        "batch_id":      batch.id,
        "file_id":       uploaded.id,
        "submitted_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_requests":    sum(1 for _ in open(BATCH_INPUT_PATH)),
    }
    BATCH_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info("메타 저장: %s", BATCH_META_PATH)
    return batch.id


def wait_and_collect(client: Any, batch_id: str) -> List[Dict[str, Any]]:
    """Batch 완료 대기 후 결과를 반환합니다."""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        logger.info("Batch 상태: %s (completed=%s, failed=%s)",
                    status,
                    batch.request_counts.completed,
                    batch.request_counts.failed)

        if status == "completed":
            break
        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} 종료: {status}")

        time.sleep(60)  # 1분마다 폴링

    content = client.files.content(batch.output_file_id).text
    results: List[Dict[str, Any]] = []
    for line in content.strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))

    logger.info("Batch 결과 수집: %d건", len(results))
    return results


def parse_batch_results(
    results: List[Dict[str, Any]],
    todo: pd.DataFrame,
) -> pd.DataFrame:
    """Batch 결과를 파싱해 todo DataFrame에 라벨을 채웁니다."""
    # custom_id → todo 행 인덱스 맵핑
    id_to_idx: Dict[str, int] = {}
    for idx, (_, row) in enumerate(todo.iterrows()):
        custom_id = f"news-{idx}-{row['event_id']}"
        id_to_idx[custom_id] = idx

    labeled = todo.copy()
    label_cols = ["macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk",
                  "primary_tag", "reasoning", "confidence"]

    # 기본값 초기화
    for col in label_cols:
        if col not in labeled.columns:
            labeled[col] = None

    success = 0
    errors  = 0

    for result in results:
        custom_id = result.get("custom_id", "")
        response  = result.get("response", {})

        if response.get("status_code") != 200:
            logger.warning("API 오류: %s → status=%s", custom_id, response.get("status_code"))
            errors += 1
            continue

        try:
            body    = response["body"]
            content = body["choices"][0]["message"]["content"]
            label   = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.warning("파싱 오류: %s → %s", custom_id, e)
            errors += 1
            continue

        # 유효성 검증
        if not _validate_label(label, custom_id):
            errors += 1
            continue

        idx = id_to_idx.get(custom_id)
        if idx is None:
            logger.warning("custom_id 매핑 실패: %s", custom_id)
            errors += 1
            continue

        row_idx = labeled.index[idx]
        labeled.at[row_idx, "macro_rate_risk"]     = label["macro_rate_risk"]
        labeled.at[row_idx, "equity_market_risk"]  = label["equity_market_risk"]
        labeled.at[row_idx, "geopolitical_fx_risk"]= label["geopolitical_fx_risk"]
        labeled.at[row_idx, "primary_tag"]         = label["primary_tag"]
        labeled.at[row_idx, "reasoning"]           = label["reasoning"]
        labeled.at[row_idx, "confidence"]          = label["confidence"]
        success += 1

    logger.info("라벨링 결과: 성공=%d, 오류=%d", success, errors)
    return labeled


def _validate_label(label: Dict[str, Any], custom_id: str) -> bool:
    """라벨 유효성 검사. 실패 시 False 반환."""
    for col in ["macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk"]:
        val = label.get(col)
        if val not in VALID_RISK_VALUES:
            logger.warning("유효하지 않은 값 [%s] %s=%s", custom_id, col, val)
            return False

    if label.get("primary_tag") not in VALID_TAGS:
        logger.warning("유효하지 않은 primary_tag [%s]: %s", custom_id, label.get("primary_tag"))
        return False

    return True


# ─────────────────────────────────────────────
# 품질 검증
# ─────────────────────────────────────────────

def validate_labels(df: pd.DataFrame) -> List[str]:
    """라벨링 품질을 검증하고 이슈 목록을 반환합니다."""
    issues: List[str] = []
    labeled = df[df["label_method"] == "llm"].dropna(
        subset=["macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk"]
    )

    if len(labeled) == 0:
        issues.append("LLM 라벨링된 이벤트 없음")
        return issues

    # 1. all-zero 비율
    all_zero_mask = (
        (labeled["macro_rate_risk"] == 0.0) &
        (labeled["equity_market_risk"] == 0.0) &
        (labeled["geopolitical_fx_risk"] == 0.0)
    )
    all_zero_rate = all_zero_mask.mean()
    if all_zero_rate > 0.7:
        issues.append(f"all-zero rate {all_zero_rate:.1%} 높음 (임계: 70%)")

    # 2. low confidence 비율
    low_conf_rate = (labeled["confidence"] < 0.66).mean()
    if low_conf_rate > 0.2:
        issues.append(f"low confidence rate {low_conf_rate:.1%} 높음 (임계: 20%)")

    # 3. 태그별 분포 출력
    print("\n=== 라벨링 분포 ===")
    print(f"전체 LLM 라벨링 건수: {len(labeled)}")
    print(f"all-zero 비율: {all_zero_rate:.1%}")
    print(f"low confidence 비율: {low_conf_rate:.1%}")
    print("\n[primary_tag 분포]")
    print(labeled["primary_tag"].value_counts())
    print("\n[macro_rate_risk 분포]")
    print(labeled["macro_rate_risk"].value_counts().sort_index())
    print("\n[equity_market_risk 분포]")
    print(labeled["equity_market_risk"].value_counts().sort_index())
    print("\n[geopolitical_fx_risk 분포]")
    print(labeled["geopolitical_fx_risk"].value_counts().sort_index())

    # 4. sanity check dates
    _run_sanity_check(df)

    return issues


SANITY_CHECKS: Dict[str, Tuple[str, str]] = {
    "2022-06-15": ("macro_rate_risk",     "FOMC 자이언트 스텝"),
    "2020-03-16": ("equity_market_risk",  "코로나 패닉"),
    "2022-02-24": ("geopolitical_fx_risk","러-우 전쟁 개전"),
    "2022-10-07": ("geopolitical_fx_risk","반도체 수출 규제"),
}


def _run_sanity_check(df: pd.DataFrame) -> None:
    print("\n=== SANITY CHECK ===")
    for date_str, (expected_tag, desc) in SANITY_CHECKS.items():
        target = pd.Timestamp(date_str)
        window = df[
            (df["date"] >= target - pd.Timedelta(days=3)) &
            (df["date"] <= target + pd.Timedelta(days=3))
        ]
        if len(window) == 0:
            print(f"  [{date_str}] {desc}: 해당 기간 이벤트 없음")
            continue

        max_val = window[expected_tag].max() if expected_tag in window.columns else None
        status = "OK" if (max_val is not None and max_val >= 0.66) else "MISS"
        print(f"  [{date_str}] {desc}: {expected_tag}={max_val} [{status}]")


# ─────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────

def save_unified(done: pd.DataFrame, labeled_todo: pd.DataFrame) -> None:
    """통합 parquet을 저장합니다."""
    unified = pd.concat([done, labeled_todo], ignore_index=True)
    unified = unified.sort_values("date").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(OUTPUT_PATH, index=False)
    logger.info("저장 완료: %s (%d건)", OUTPUT_PATH, len(unified))


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main(submit_only: bool = False,
         collect_only: bool = False,
         validate: bool = False) -> None:

    if validate:
        if not OUTPUT_PATH.exists():
            logger.error("unified parquet 없음. 먼저 전체 파이프라인을 실행하세요.")
            return
        df = pd.read_parquet(OUTPUT_PATH)
        df["date"] = pd.to_datetime(df["date"])
        issues = validate_labels(df)
        if issues:
            print("\n⚠️  품질 이슈:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ 품질 검증 통과")
        return

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except ImportError:
        raise ImportError("openai 패키지가 없습니다. pip install openai 실행 후 재시도.")

    if collect_only:
        if not BATCH_META_PATH.exists():
            logger.error("배치 메타 파일 없음: %s", BATCH_META_PATH)
            return
        meta = json.loads(BATCH_META_PATH.read_text())
        batch_id = meta["batch_id"]
        logger.info("기존 Batch 결과 수집: %s", batch_id)

        all_events = load_all_events()
        if all_events.empty:
            return
        done, todo = split_events(all_events)

        results = wait_and_collect(client, batch_id)
        labeled_todo = parse_batch_results(results, todo)
        save_unified(done, labeled_todo)

        issues = validate_labels(pd.concat([done, labeled_todo], ignore_index=True))
        if issues:
            logger.warning("품질 이슈: %s", issues)
        return

    # 전체 파이프라인
    all_events = load_all_events()
    if all_events.empty:
        logger.error("처리할 이벤트가 없습니다.")
        return

    done, todo = split_events(all_events)

    if len(todo) == 0:
        logger.info("LLM 라벨링 대상 없음. 저장만 진행합니다.")
        save_unified(done, pd.DataFrame())
        return

    build_batch_jsonl(todo)

    if submit_only:
        logger.info("--submit-only: Batch 제출 후 종료합니다.")
        submit_batch(client)
        logger.info("결과 수집은 나중에: python scripts/label_events_llm.py --collect-only")
        return

    # 제출 + 대기 + 수집
    batch_id = submit_batch(client)
    results  = wait_and_collect(client, batch_id)
    labeled_todo = parse_batch_results(results, todo)
    save_unified(done, labeled_todo)

    issues = validate_labels(pd.concat([done, labeled_todo], ignore_index=True))
    if issues:
        logger.warning("품질 이슈 발견:\n%s", "\n".join(f"  - {i}" for i in issues))
    else:
        logger.info("✅ 라벨링 완료 + 품질 검증 통과")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 배치 라벨링 파이프라인")
    parser.add_argument("--submit-only",  action="store_true",
                        help="Batch 제출만 하고 종료 (결과는 나중에 --collect-only로 수집)")
    parser.add_argument("--collect-only", action="store_true",
                        help="기존 batch_id로 결과만 수집 (submit 없이)")
    parser.add_argument("--validate",     action="store_true",
                        help="저장된 unified parquet 품질 검증만 실행")
    args = parser.parse_args()

    main(
        submit_only  = args.submit_only,
        collect_only = args.collect_only,
        validate     = args.validate,
    )
