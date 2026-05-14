from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.rl.env import PortfolioEnv


def main():
    returns_df = pd.read_parquet("data/processed/returns.parquet")
    features_df = pd.read_parquet("data/processed/features.parquet")

    env = PortfolioEnv(
        returns_df=returns_df,
        features_df=features_df,
        lookback=30,
        reward_type="return",
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
        total_timesteps=50_000,
        tb_log_name="ppo_return_50k",
    )

    Path("models").mkdir(exist_ok=True)
    model.save("models/ppo_return_50k_risk")

    print("PPO 50k 학습 완료")
    print("모델 저장: models/ppo_return_50k_risk.zip")


if __name__ == "__main__":
    main()