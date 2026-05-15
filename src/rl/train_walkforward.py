import json
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.rl.env import PortfolioEnv


RETURNS_PATH = Path("data/processed/returns.parquet")
RAW_FEATURES_PATH = Path("data/processed/raw_features.parquet")
SCALERS_DIR = Path("data/processed/scalers")
MODELS_DIR = Path("models")
TENSORBOARD_DIR = Path("logs/tensorboard")

LOOKBACK = 30
TOTAL_TIMESTEPS = 100_000

WINDOWS = {
    "w1": {
        "train_start": "2018-01-01",
        "train_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2022-12-31",
    },
    "w2": {
        "train_start": "2019-01-01",
        "train_end": "2022-12-31",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31",
    },
    "w3": {
        "train_start": "2020-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
    "final": {
        "train_start": "2021-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
}

REWARD_TYPES = ["return", "sharpe", "mdd"]


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """학습에 사용할 returns/raw_features parquet 파일을 한 번만 로드합니다.

    Returns:
        (returns_df, raw_features_df) — 둘 다 정규화 전 raw 값.
    """
    returns_df = pd.read_parquet(RETURNS_PATH)
    raw_features_df = pd.read_parquet(RAW_FEATURES_PATH)

    returns_df.index = pd.to_datetime(returns_df.index)
    raw_features_df.index = pd.to_datetime(raw_features_df.index)

    returns_df = returns_df.sort_index()
    raw_features_df = raw_features_df.sort_index()

    return returns_df, raw_features_df


def normalize_features(
    train_features: pd.DataFrame,
    window_name: str,
) -> tuple[pd.DataFrame, dict]:
    """학습 구간 통계로 Z-score 정규화하고 통계를 저장합니다.

    Args:
        train_features: 학습 구간 raw feature DataFrame.
        window_name: 윈도우 식별자 (저장 경로에 사용).

    Returns:
        (정규화된 train_features, {"mean": {...}, "std": {...}}) 튜플.
        std가 0인 컬럼은 1로 대체하여 ZeroDivisionError를 방지합니다.
    """
    mean = train_features.mean()
    std = train_features.std().replace(0, 1)

    normalized = (train_features - mean) / std

    stats = {
        "mean": mean.to_dict(),
        "std": std.to_dict(),
    }
    stats_path = SCALERS_DIR / f"{window_name}_feature_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"정규화 통계 저장: {stats_path}")
    return normalized, stats


def apply_normalization(
    features: pd.DataFrame,
    stats: dict,
) -> pd.DataFrame:
    """저장된 train 통계로 임의 구간 feature를 Z-score 변환합니다.

    Args:
        features: 변환할 raw feature DataFrame.
        stats: normalize_features()가 반환한 {"mean": ..., "std": ...} dict.

    Returns:
        정규화된 feature DataFrame.
    """
    mean = pd.Series(stats["mean"])
    std = pd.Series(stats["std"])
    return (features - mean) / std


def align_data(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    window_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """returns/features의 공통 날짜만 사용하도록 정렬합니다.

    Args:
        returns: 수익률 DataFrame.
        features: 피처 DataFrame.
        window_name: 오류 메시지용 윈도우 이름.

    Returns:
        (aligned_returns, aligned_features) 튜플.

    Raises:
        ValueError: 공통 날짜 수가 lookback 이하인 경우.
    """
    common_index = returns.index.intersection(features.index)

    if len(common_index) <= LOOKBACK:
        raise ValueError(
            f"{window_name} has insufficient aligned data: "
            f"{len(common_index)} rows"
        )

    return returns.loc[common_index].copy(), features.loc[common_index].copy()


def train_one_model(
    returns_df: pd.DataFrame,
    raw_features_df: pd.DataFrame,
    window_name: str,
    window: dict[str, str],
    reward_type: str,
    total_timesteps: int = TOTAL_TIMESTEPS,
) -> None:
    """단일 reward/window 조합의 PPO 모델을 학습하고 저장합니다.

    학습 구간 raw_features에서 mean/std를 계산하여 Z-score 정규화한 뒤
    PortfolioEnv에 전달합니다. 정규화 통계는 data/processed/scalers/에 저장합니다.

    Args:
        returns_df: 전체 기간 raw 로그수익률.
        raw_features_df: 전체 기간 정규화 전 raw feature.
        window_name: 윈도우 식별자 ("w1" | "w2" | "w3" | "final").
        window: 윈도우 날짜 정의 dict.
        reward_type: 보상함수 종류 ("return" | "sharpe" | "mdd").
        total_timesteps: PPO 총 학습 스텝 수.
    """
    # 학습 구간 분리
    train_returns = returns_df.loc[window["train_start"]:window["train_end"]].copy()
    train_raw = raw_features_df.loc[window["train_start"]:window["train_end"]].copy()

    # 공통 날짜 정렬
    train_returns, train_raw = align_data(train_returns, train_raw, window_name)

    # 학습 구간 통계로 Z-score 정규화 + 통계 저장
    train_features, _ = normalize_features(train_raw, window_name)

    env = PortfolioEnv(
        returns_df=train_returns,
        features_df=train_features,
        lookback=LOOKBACK,
        reward_type=reward_type,
    )
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        tensorboard_log=str(TENSORBOARD_DIR),
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=f"ppo_{reward_type}_{window_name}",
    )

    model_path = MODELS_DIR / f"ppo_{reward_type}_{window_name}_risk"
    model.save(str(model_path))

    print(f"[완료] {reward_type} / {window_name}")
    print(f"모델 저장: {model_path}.zip")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)

    returns_df, raw_features_df = load_training_data()

    for reward_type in REWARD_TYPES:
        for window_name, window in WINDOWS.items():
            train_one_model(
                returns_df=returns_df,
                raw_features_df=raw_features_df,
                window_name=window_name,
                window=window,
                reward_type=reward_type,
                total_timesteps=TOTAL_TIMESTEPS,
            )


if __name__ == "__main__":
    main()
