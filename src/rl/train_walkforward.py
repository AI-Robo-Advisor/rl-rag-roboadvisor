from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.rl.env import PortfolioEnv


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


def split_train_data(df: pd.DataFrame, window: dict) -> pd.DataFrame:
    return df.loc[window["train_start"]:window["train_end"]].copy()


def train_one_model(
    window_name: str,
    window: dict,
    reward_type: str,
    total_timesteps: int = 100_000,
) -> None:
    returns_df = pd.read_parquet("data/processed/returns.parquet")
    features_df = pd.read_parquet("data/processed/features.parquet")

    returns_df.index = pd.to_datetime(returns_df.index)
    features_df.index = pd.to_datetime(features_df.index)

    returns_df = returns_df.sort_index()
    features_df = features_df.sort_index()

    train_returns = split_train_data(returns_df, window)
    train_features = split_train_data(features_df, window)

    env = PortfolioEnv(
        returns_df=train_returns,
        features_df=train_features,
        lookback=30,
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
        tensorboard_log="logs/tensorboard",
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=f"ppo_{reward_type}_{window_name}",
    )

    Path("models").mkdir(exist_ok=True)

    model_path = f"models/ppo_{reward_type}_{window_name}"
    model.save(model_path)

    print(f"[완료] {reward_type} / {window_name}")
    print(f"모델 저장: {model_path}.zip")


def main():
    for reward_type in REWARD_TYPES:
        for window_name, window in WINDOWS.items():
            train_one_model(
                window_name=window_name,
                window=window,
                reward_type=reward_type,
                total_timesteps=100_000,
            )


if __name__ == "__main__":
    main()