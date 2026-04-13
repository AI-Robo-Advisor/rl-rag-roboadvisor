## 관련 이슈
closes #

## 작업 내용
이문정(m) 담당 RL 환경 기초 구현 — Sprint 1 범위.

- `scripts/cartpole_ppo.py`: CartPole-v1 환경에서 PPO 학습 동작 확인용 예제 스크립트 작성 (SB3 연동 검증)
- `src/rl/env.py`: 실제 포트폴리오 최적화에 쓸 `PortfolioEnv` 클래스 뼈대 구현 (Gymnasium `gym.Env` 상속)
- `requirements.txt`: SB3 PPO 학습 의존성 추가 및 전체 버전 고정

## 변경된 파일
| 파일 | 변경 내용 |
|------|---------|
| `scripts/cartpole_ppo.py` | CartPole-v1 + PPO 학습·테스트 예제 (10,000 스텝 학습, deterministic 평가 루프) |
| `src/rl/env.py` | `PortfolioEnv` 클래스 뼈대 — `observation_space`, `action_space`, `reset`, `step`, `render` 구현 |
| `requirements.txt` | `gymnasium[classic-control]==1.2.3`, `stable-baselines3==2.8.0`, `numpy==2.2.6`, `pytest==9.0.3` 추가 및 기존 패키지 버전 고정 |

## 구현 세부 사항

### PortfolioEnv 설계
- **관찰 공간**: `lookback × n_assets` 수익률 윈도우 + 현재 포트폴리오 비중 (연속, `-inf ~ inf`)
- **행동 공간**: 자산별 비중 `[0.0, 1.0]` → 내부에서 합계가 1이 되도록 정규화
- **보상**: 한 스텝의 포트폴리오 가중 수익률 (`dot(weights, asset_returns)`)
- **종료 조건**: 데이터 끝 도달 시 `terminated=True`
- **Sprint 1 제약**: features_df에 RSI·MACD 컬럼 필요 (아직 `src/data/indicators.py` 미연동, 더미 데이터로 단위 테스트 예정)

## 테스트 결과
```
# CartPole PPO 동작 확인 (수동 실행)
python scripts/cartpole_ppo.py
# → SB3 PPO 학습 10,000 스텝 정상 완료
# → deterministic 평가에서 episode score 출력 확인

# pytest — 현재 tests/ 없음, Sprint 2에서 PortfolioEnv 단위 테스트 추가 예정
pytest tests/ -v  # (추후 실행)
```

## 체크리스트
- [x] 담당 파일 외 다른 팀원 파일 무단 수정 없음
- [x] `.env` 파일 커밋 안 함
- [ ] 새 함수/클래스에 docstring 작성 — `PortfolioEnv` docstring Sprint 2에서 보완 예정
- [ ] 테스트 통과 확인 (`pytest`) — `tests/` 디렉터리 미생성, Sprint 2에서 추가 예정
- [x] PR 대상 브랜치가 `dev`인지 확인 (셀프 merge 금지)

## 기타 참고 사항
- `PortfolioEnv._get_observation()`에서 `features_df`의 RSI·MACD 컬럼을 직접 참조하므로, 박지민(j)의 `src/data/indicators.py` 구현 완료 후 실데이터 연동 필요합니다.
- `requirements.txt` 버전 고정으로 팀 전체 환경이 영향받을 수 있으니 충돌 발생 시 알려주세요.
- CartPole 예제(`scripts/cartpole_ppo.py`)는 SB3·Gymnasium 연동 검증용으로, 실서비스 코드가 아닙니다.
