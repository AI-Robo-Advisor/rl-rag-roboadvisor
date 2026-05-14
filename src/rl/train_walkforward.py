from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.rl.env import PortfolioEnv


RETURNS_PATH = Path("data/processed/returns.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
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
    """학습에 사용할 returns/features parquet 파일을 한 번만 로드합니다."""
    returns_df = pd.read_parquet(RETURNS_PATH)
    features_df = pd.read_parquet(FEATURES_PATH)

    returns_df.index = pd.to_datetime(returns_df.index)
    features_df.index = pd.to_datetime(features_df.index)

    returns_df = returns_df.sort_index()
    features_df = features_df.sort_index()

    return returns_df, features_df


def split_train_data(df: pd.DataFrame, window: dict[str, str]) -> pd.DataFrame:
    """Walk-Forward window의 train 구간만 분리합니다."""
    return df.loc[window["train_start"]:window["train_end"]].copy()


def align_train_data(
    train_returns: pd.DataFrame,
    train_features: pd.DataFrame,
    window_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """returns/features의 공통 날짜만 사용하도록 정렬합니다."""
    common_index = train_returns.index.intersection(train_features.index)

    if len(common_index) <= LOOKBACK:
        raise ValueError(
            f"{window_name} has insufficient aligned data: "
            f"{len(common_index)} rows"
        )

    aligned_returns = train_returns.loc[common_index].copy()
    aligned_features = train_features.loc[common_index].copy()

    return aligned_returns, aligned_features


def train_one_model(
    returns_df: pd.DataFrame,
    features_df: pd.DataFrame,
    window_name: str,
    window: dict[str, str],
    reward_type: str,
    total_timesteps: int = TOTAL_TIMESTEPS,
) -> None:
    """단일 reward/window 조합의 PPO 모델을 학습하고 저장합니다."""
    train_returns = split_train_data(returns_df, window)
    train_features = split_train_data(features_df, window)

    train_returns, train_features = align_train_data(
        train_returns=train_returns,
        train_features=train_features,
        window_name=window_name,
    )

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

    model_path = MODELS_DIR / f"ppo_{reward_type}_{window_name}"
    model.save(str(model_path))

    print(f"[완료] {reward_type} / {window_name}")
    print(f"모델 저장: {model_path}.zip")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    returns_df, features_df = load_training_data()

    for reward_type in REWARD_TYPES:
        for window_name, window in WINDOWS.items():
            train_one_model(
                returns_df=returns_df,
                features_df=features_df,
                window_name=window_name,
                window=window,
                reward_type=reward_type,
                total_timesteps=TOTAL_TIMESTEPS,
            )


if __name__ == "__main__":
    main()