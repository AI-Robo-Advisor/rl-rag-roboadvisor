"""SHAP explainability module for PPO portfolio model.

담당: 이문정 (src/rl/shap.py)

PPO 정책의 의사결정을 SHAP으로 해석하고, /explain 엔드포인트 연동용
인터페이스를 제공합니다. apps/api/services.py(박지민)가 아래 함수를
호출하는 방식으로 통합됩니다.

공개 인터페이스:
    is_shap_ready(model_path)                → bool        (모델·패키지 로드 가능 여부)
    get_feature_names(asset_names, lookback) → list[str]   (333개, 모델 불필요)
    compute_shap_explanation(...)            → dict         (ExplainResponse 호환)
    generate_summary_plot(...)              → None          (파일 또는 화면 출력)
    generate_force_plot(...)               → None           (파일 또는 화면 출력)

────────────────────────────────────────────────────────────
박지민(services.py) 연동 가이드
────────────────────────────────────────────────────────────

1) build_module_statuses() — "shap" 상태 판단:

    from src.rl.shap import is_shap_ready, DEFAULT_MODEL_PATH

    "shap": "ready" if is_shap_ready(DEFAULT_MODEL_PATH) else "fallback"

2) build_shap_explanation() — 실제 SHAP 반환:

    from src.rl.shap import compute_shap_explanation, DEFAULT_MODEL_PATH

    try:
        features_df = pd.read_parquet("data/processed/features.parquet")
        returns_df  = pd.read_parquet("data/processed/returns.parquet")
        result = compute_shap_explanation(
            model_path=DEFAULT_MODEL_PATH,
            features_df=features_df,
            returns_df=returns_df,
            date=date,        # ExplainRequest.date
            top_k=top_k,      # ExplainRequest.top_k
        )
        return ExplainResponse(
            status="ready",
            date=date,
            target_date=result["target_date"],
            base_value=result["base_value"],
            prediction=result["prediction"],
            feature_contributions=[
                FeatureContribution(**fc) for fc in result["feature_contributions"]
            ],
            feature_names=result["feature_names"],
            shap_values=result["shap_values"],
            message="PPO SHAP 분석 완료.",
        )
    except Exception:
        return build_fallback_explanation(date, top_k)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# RL 리스크 태그 피처명 (관측공간 마지막 3개, labels_and_interfaces.md 2-2절 참고)
RISK_FEATURE_NAMES: list[str] = ["risk_규제변경", "risk_실적쇼크", "risk_급등락"]

DEFAULT_ASSET_NAMES: list[str] = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "TLT", "GLD", "VNQ", "069500", "114260",
]

# 기본 모델 경로 (labels_and_interfaces.md 3-3절, reward_analysis_report.md 8절 참고)
# sharpe 보상이 4개 윈도우 전체에서 누적수익률·샤프비율·알파 최상위
# _risk 접미사: obs_dim=333 (risk_vector 포함) 환경으로 학습된 모델
DEFAULT_MODEL_PATH: Path = Path("models/ppo_sharpe_final_risk.zip")


def is_shap_ready(model_path: str | Path = DEFAULT_MODEL_PATH) -> bool:
    """SHAP 모듈과 PPO 모델이 사용 가능한지 확인합니다.

    services.py의 build_module_statuses()에서 "shap" 상태를 판단할 때 사용합니다.

    Args:
        model_path: 확인할 PPO 모델 파일 경로. 기본값은 DEFAULT_MODEL_PATH.

    Returns:
        True: shap·stable-baselines3·torch 모두 import 가능하고
              model_path 파일이 존재할 때.
        False: 그 외 모든 경우.
    """
    if not Path(model_path).exists():
        return False
    try:
        import shap  # noqa: F401
        import torch  # noqa: F401
        from stable_baselines3 import PPO  # noqa: F401
        return True
    except ImportError:
        return False


def get_feature_names(asset_names: list[str], lookback: int = 30) -> list[str]:
    """PortfolioEnv 관측공간의 피처명 목록을 반환합니다.

    _get_observation() concat 순서와 정확히 일치합니다:
        [returns_window | weights | RSI | MACD_signal | risk_vector]

    관측 차원 = (lookback + 3) × n_assets + 3

    수익률 윈도우 lag 규칙:
        t-{lookback} = 가장 오래된 관측 (current_step - lookback)
        t-1          = 가장 최근 관측 (current_step - 1)

    SHAP feature_names와 shap_values 인덱스가 1:1 대응하므로
    이 함수의 순서를 바꾸면 SHAP 해석이 어긋납니다.

    Args:
        asset_names: 자산 티커 목록. PortfolioEnv.asset_names와 동일한 순서.
        lookback: 과거 수익률 윈도우 길이 (기본값 30).

    Returns:
        길이 = (lookback + 3) × len(asset_names) + 3 인 피처명 리스트.
    """
    names: list[str] = []

    # 수익률 윈도우: lookback × n_assets (row-major, 오래된 것부터)
    # features_df[return_cols].iloc[step-lookback : step].values.flatten()
    for lag in range(lookback, 0, -1):          # t-lookback(oldest) → t-1(newest)
        for asset in asset_names:
            names.append(f"{asset}_return_t-{lag}")

    # 현재 포트폴리오 비중: n_assets
    for asset in asset_names:
        names.append(f"weight_{asset}")

    # RSI(14): n_assets
    for asset in asset_names:
        names.append(f"{asset}_RSI")

    # MACD signal: n_assets
    for asset in asset_names:
        names.append(f"{asset}_MACD_signal")

    # RAG 리스크 태그 벡터: 3 (반드시 마지막)
    names.extend(RISK_FEATURE_NAMES)

    return names


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _resolve_target_step(
    features_df: pd.DataFrame,
    date: str | None,
    lookback: int,
) -> tuple[int, str]:
    """날짜 문자열로부터 target_step 인덱스와 날짜 문자열을 반환합니다.

    Args:
        features_df: 정렬된 피처 DataFrame.
        date: 기준 날짜 (YYYY-MM-DD). None이면 마지막 행.
        lookback: 최소 step 하한.

    Returns:
        (step_index, date_string) 튜플.
    """
    if date is not None:
        ts = pd.Timestamp(date)
        candidates = features_df.loc[features_df.index <= ts]
        idx = len(candidates) - 1 if not candidates.empty else lookback
    else:
        idx = len(features_df) - 1
    idx = max(idx, lookback)
    return idx, features_df.index[idx].strftime("%Y-%m-%d")


def _collect_obs(env: Any, steps: range) -> np.ndarray:
    """지정 step 범위에서 observation을 수집합니다.

    Args:
        env: PortfolioEnv 인스턴스.
        steps: 수집할 step 범위.

    Returns:
        shape = (len(steps), obs_dim) 의 float32 배열.
    """
    obs_list = []
    for step in steps:
        env.current_step = step
        obs_list.append(env._get_observation())
    return np.array(obs_list, dtype=np.float32)


def _make_value_predict_fn(model: Any) -> Any:
    """PPO 가치함수 V(obs)를 SHAP용 callable로 감쌉니다.

    V(obs)는 해당 관측 상태에서 기대되는 누적 보상의 추정값입니다.
    'obs에서 어떤 피처가 에이전트의 가치 판단에 기여하는가'를 설명하므로
    RL SHAP explainability에 적합합니다.

    Args:
        model: SB3 PPO 모델.

    Returns:
        obs_batch (np.ndarray, shape=(n, obs_dim)) → values (np.ndarray, shape=(n,))
    """
    import torch as th

    def predict_fn(obs_batch: np.ndarray) -> np.ndarray:
        results = []
        for obs in obs_batch:
            tensor_obs = th.FloatTensor(obs).unsqueeze(0).to(model.device)
            with th.no_grad():
                value = model.policy.predict_values(tensor_obs)
            val = float(value.cpu().numpy().item())
            # KernelExplainer의 perturbation이 obs 분포 밖 샘플을 생성할 때
            # 가치함수가 발산하는 것을 방지하기 위한 클리핑
            val = float(np.clip(val, -1000.0, 1000.0))
            results.append(val)
        return np.array(results, dtype=np.float64)

    return predict_fn


# ---------------------------------------------------------------------------
# 공개 인터페이스
# ---------------------------------------------------------------------------

def compute_shap_explanation(
    model_path: str | Path,
    features_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    date: str | None = None,
    top_k: int = 8,
    risk_vector: np.ndarray | None = None,
    lookback: int = 30,
    background_size: int = 50,
    nsamples: int = 100,
) -> dict[str, Any]:
    """PPO 모델의 특정 날짜 의사결정에 대한 SHAP 설명을 계산합니다.

    KernelExplainer(모델 독립적)로 SHAP 값을 계산하여
    ExplainResponse 스키마 호환 dict를 반환합니다.

    Args:
        model_path: 저장된 PPO 모델 경로 (.zip).
        features_df: 정규화된 피처 DataFrame (features.parquet).
        returns_df: raw 로그수익률 DataFrame (returns.parquet).
        date: 설명 기준 날짜 (YYYY-MM-DD). None이면 최신 날짜.
        top_k: 반환할 상위 피처 수 (|SHAP| 내림차순).
        risk_vector: 3종 리스크 태그 벡터. None이면 zeros(3).
        lookback: 관측 윈도우 길이 (학습 시 사용한 값과 동일해야 함).
        background_size: KernelExplainer 배경 샘플 수.
        nsamples: KernelExplainer perturbation 샘플 수.

    Returns:
        ExplainResponse 스키마와 1:1 대응하는 dict:
            - status (str): "ready"
            - date (str | None): 요청 날짜 (입력 그대로 echo)
            - target_date (str): 실제 사용된 데이터 날짜
            - base_value (float): SHAP base value (모델 평균 예측값)
            - prediction (float): base_value + sum(shap_values)
            - feature_contributions (list[dict]):
                  각 dict = {"feature": str, "value": float, "contribution": float}
            - feature_names (list[str]): top_k 피처명 (|SHAP| 내림차순)
            - shap_values (list[float]): top_k SHAP 값
            - message (str)

    Raises:
        FileNotFoundError: model_path가 존재하지 않을 때.
        ImportError: shap 또는 stable-baselines3가 설치되지 않았을 때.
    """
    import shap
    from stable_baselines3 import PPO
    from src.rl.env import PortfolioEnv

    model = PPO.load(str(model_path))

    common_index = features_df.index.intersection(returns_df.index)
    features_aligned = features_df.loc[common_index]
    returns_aligned = returns_df.loc[common_index]

    target_step, target_date = _resolve_target_step(features_aligned, date, lookback)

    env = PortfolioEnv(
        returns_df=returns_aligned,
        features_df=features_aligned,
        lookback=lookback,
        reward_type="sharpe",
        risk_vector=risk_vector,
    )

    feature_names = get_feature_names(env.asset_names, lookback)

    # 배경 데이터: target_step 직전 background_size개 관측치
    bg_start = max(lookback, target_step - background_size)
    background = _collect_obs(env, range(bg_start, target_step))

    # 설명 대상 관측치
    env.current_step = target_step
    target_obs = env._get_observation()

    predict_fn = _make_value_predict_fn(model)
    explainer = shap.KernelExplainer(predict_fn, background)
    raw_shap = explainer.shap_values(
        target_obs.reshape(1, -1), nsamples=nsamples, silent=True
    )
    shap_values_arr = np.array(raw_shap).flatten()

    base_value = float(explainer.expected_value)
    prediction = float(base_value + shap_values_arr.sum())

    # |SHAP| 내림차순 top_k 추출
    top_indices = np.argsort(np.abs(shap_values_arr))[::-1][:top_k]

    feature_contributions = [
        {
            "feature": feature_names[i],
            "value": round(float(target_obs[i]), 6),
            "contribution": round(float(shap_values_arr[i]), 6),
        }
        for i in top_indices
    ]

    return {
        "status": "ready",
        "date": date,
        "target_date": target_date,
        "base_value": round(base_value, 6),
        "prediction": round(prediction, 6),
        "feature_contributions": feature_contributions,
        "feature_names": [fc["feature"] for fc in feature_contributions],
        "shap_values": [fc["contribution"] for fc in feature_contributions],
        "message": f"PPO SHAP 분석 완료 (기준일: {target_date}, top_k={top_k}).",
    }


def generate_summary_plot(
    model_path: str | Path,
    features_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    risk_vector: np.ndarray | None = None,
    lookback: int = 30,
    n_samples: int = 100,
    background_size: int = 50,
    nsamples_per_obs: int = 100,
    save_path: str | Path | None = None,
) -> None:
    """SHAP Summary Plot을 생성합니다.

    n_samples개 관측치에 대한 피처 중요도 분포를 시각화합니다.
    피처 순서는 get_feature_names() 와 동일하므로 SHAP 해석이 정확합니다.

    Args:
        model_path: 저장된 PPO 모델 경로 (.zip).
        features_df: 정규화된 피처 DataFrame.
        returns_df: raw 로그수익률 DataFrame.
        risk_vector: 3종 리스크 태그 벡터. None이면 zeros(3).
        lookback: 관측 윈도우 길이.
        n_samples: 시각화에 사용할 관측 샘플 수.
        background_size: KernelExplainer 배경 샘플 수.
        nsamples_per_obs: 관측당 perturbation 샘플 수.
        save_path: 저장 경로 (PNG). None이면 화면 출력.

    Raises:
        ImportError: shap 또는 stable-baselines3가 설치되지 않았을 때.
    """
    import shap
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO
    from src.rl.env import PortfolioEnv

    model = PPO.load(str(model_path))

    common_index = features_df.index.intersection(returns_df.index)
    features_aligned = features_df.loc[common_index]
    returns_aligned = returns_df.loc[common_index]

    env = PortfolioEnv(
        returns_df=returns_aligned,
        features_df=features_aligned,
        lookback=lookback,
        reward_type="sharpe",
        risk_vector=risk_vector,
    )

    feature_names = get_feature_names(env.asset_names, lookback)

    n_total = len(features_aligned) - lookback
    step_size = max(1, n_total // n_samples)
    sample_steps = list(range(lookback, len(features_aligned), step_size))[:n_samples]

    obs_matrix = _collect_obs(env, sample_steps)
    background = obs_matrix[: min(background_size, len(obs_matrix))]

    predict_fn = _make_value_predict_fn(model)
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_matrix = explainer.shap_values(
        obs_matrix, nsamples=nsamples_per_obs, silent=True
    )

    shap.summary_plot(
        shap_matrix,
        obs_matrix,
        feature_names=feature_names,
        show=save_path is None,
    )

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Summary Plot 저장 완료: {save_path}")


def generate_force_plot(
    model_path: str | Path,
    features_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    date: str | None = None,
    risk_vector: np.ndarray | None = None,
    lookback: int = 30,
    background_size: int = 50,
    nsamples: int = 100,
    save_path: str | Path | None = None,
) -> None:
    """SHAP Force Plot을 HTML로 생성합니다 (단일 관측 설명).

    특정 날짜의 관측값에 대해 각 피처가 모델의 가치 판단에
    어떻게 기여했는지를 인터랙티브 HTML로 저장합니다.

    저장 경로 기본값: data/results/shap_force_<target_date>.html

    Args:
        model_path: 저장된 PPO 모델 경로 (.zip).
        features_df: 정규화된 피처 DataFrame.
        returns_df: raw 로그수익률 DataFrame.
        date: 설명 기준 날짜 (YYYY-MM-DD). None이면 최신.
        risk_vector: 3종 리스크 태그 벡터. None이면 zeros(3).
        lookback: 관측 윈도우 길이.
        background_size: KernelExplainer 배경 샘플 수.
        nsamples: perturbation 샘플 수.
        save_path: 저장 경로 (.html). None이면 data/results/shap_force_<date>.html.

    Raises:
        ImportError: shap 또는 stable-baselines3가 설치되지 않았을 때.
    """
    import shap
    from stable_baselines3 import PPO
    from src.rl.env import PortfolioEnv

    model = PPO.load(str(model_path))

    common_index = features_df.index.intersection(returns_df.index)
    features_aligned = features_df.loc[common_index]
    returns_aligned = returns_df.loc[common_index]

    target_step, target_date = _resolve_target_step(features_aligned, date, lookback)

    env = PortfolioEnv(
        returns_df=returns_aligned,
        features_df=features_aligned,
        lookback=lookback,
        reward_type="sharpe",
        risk_vector=risk_vector,
    )

    feature_names = get_feature_names(env.asset_names, lookback)

    bg_start = max(lookback, target_step - background_size)
    background = _collect_obs(env, range(bg_start, target_step))

    env.current_step = target_step
    target_obs = env._get_observation()

    predict_fn = _make_value_predict_fn(model)
    explainer = shap.KernelExplainer(predict_fn, background)
    raw_shap = explainer.shap_values(
        target_obs.reshape(1, -1), nsamples=nsamples, silent=True
    )
    shap_values = np.array(raw_shap).flatten()

    # Force Plot은 HTML(인터랙티브)로 저장
    force = shap.force_plot(
        explainer.expected_value,
        shap_values,
        features=target_obs,
        feature_names=feature_names,
        matplotlib=False,
    )

    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    html_path = Path(save_path) if save_path is not None else (
        results_dir / f"shap_force_{target_date}.html"
    )
    shap.save_html(str(html_path), force)
    print(f"Force Plot 저장 완료: {html_path}")
