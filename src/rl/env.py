import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    def __init__(self, returns_df, features_df, lookback=30):
        super().__init__()

        self.returns_df = returns_df
        self.features_df = features_df
        self.lookback = lookback

        self.asset_names = list(returns_df.columns)
        self.n_assets = len(self.asset_names)

        self.current_step = self.lookback
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        # Sprint1에서는 returns window + current weights만 사용
        obs_dim = (self.lookback + 3) * self.n_assets

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

    def _get_observation(self):
        returns_window = self.returns_df.iloc[
            self.current_step - self.lookback:self.current_step
        ].values

        rsi = self.features_df[[f"{asset}_RSI" for asset in self.asset_names]].iloc[
            self.current_step
        ].values

        macd = self.features_df[[f"{asset}_MACD" for asset in self.asset_names]].iloc[
            self.current_step
        ].values

        obs = np.concatenate([
            returns_window.flatten(),
            self.weights,
            rsi,
            macd,
        ]).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.lookback
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)

        if action.sum() < 1e-6:
            action = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        else:
            action = action / action.sum()

        self.weights = action

        asset_returns = self.returns_df.iloc[self.current_step].values.astype(np.float32)
        reward = float(np.dot(self.weights, asset_returns))

        self.current_step += 1

        terminated = self.current_step >= len(self.returns_df)
        truncated = False

        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._get_observation()

        info = {
            "weights": self.weights,
            "step": self.current_step,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        print(
            f"step={self.current_step}, "
            f"weights={np.round(self.weights, 3)}"
        )