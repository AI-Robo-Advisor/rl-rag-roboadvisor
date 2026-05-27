# risk_signals handoff (Agent -> API -> RL)

## 목적

`/research`에서 생성한 정량 위험 신호를 `/optimize`로 전달해 RL 관측 벡터에 반영한다.

## 공통 계약

- 필드명: `risk_signals`
- 타입: `list[dict]`
- 원소 스키마:
  - `tag`: `"macro_rate_risk" | "equity_market_risk" | "geopolitical_fx_risk"`
  - `severity`: `float` (`0.0 ~ 1.0`)

예시:

```json
[
  {"tag": "macro_rate_risk", "severity": 1.0},
  {"tag": "equity_market_risk", "severity": 0.66}
]
```

## Agent 구현 지점

- 파일: `src/agent/nodes.py`
- 노드: `analyst_node`
- 입력: 검색 문서 본문 결합 텍스트
- 계산: `score_risk_vector(all_text)` -> `(macro, equity, geo)`
- 출력:
  - `rl_risk_tags` (기존 유지)
  - `risk_signals` (신규)

## API 처리

- 파일: `apps/api/services.py`
- `/research`:
  - `state["risk_signals"]`가 있으면 유효한 `tag`와 `0.0 <= severity <= 1.0` 행만 정규화해 응답
  - 없으면 `risk_tags`를 `severity=1.0`으로 승격해 fallback
- `/optimize`:
  - `risk_signals` 우선
  - 없으면 legacy `risk_tags` 승격
  - 둘 다 없으면 `PortfolioEnv` 기본 동작(파케 자동 로드)
  - 신호가 있으면 `PortfolioEnv` 생성자에 `risk_vector`를 직접 넘겨 불필요한 파케 자동 로드를 피함

## RL 벡터 변환

- 함수: `_signals_to_vector(signals)`
- 변환:
  - 고정 순서: `["macro_rate_risk", "equity_market_risk", "geopolitical_fx_risk"]`
  - `apply_decay(severity, 0, tag)` 사용
- 결과 dtype: `np.float32`

## Dashboard 전달

- `complete`/`fallback` 이벤트에서 `risk_signals`를 세션에 저장
- `/optimize` 요청 payload에 `risk_signals` 포함
