import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    def __init__(
        self, 
        returns_df, 
        features_df, 
        lookback=30,
        reward_type="return",
        lambda_mdd=1.0,
        volatility_window=20,
        ):
        super().__init__()

        valid_reward_types = {"return", "sharpe", "mdd"}
        if reward_type not in valid_reward_types:
            raise ValueError(
                f"reward_type must be one of {valid_reward_types}, got {reward_type}"
            )

        self.lookback = lookback
        self.reward_type = reward_type
        self.lambda_mdd = lambda_mdd
        self.volatility_window = volatility_window

        # features 기준 날짜로 returns 정렬
        common_index = features_df.index.intersection(returns_df.index)
        self.features_df = features_df.loc[common_index].copy()
        self.returns_df = returns_df.loc[common_index].copy()

        self.asset_names = list(self.returns_df.columns)
        self.n_assets = len(self.asset_names)

        self.fee_rate = 0.00015       # 거래비용 0.015%
        self.slippage_rate = 0.0005   # 슬리피지 0.05%
        self.total_cost_rate = self.fee_rate + self.slippage_rate

        self.safe_guard_mdd = 0.15

        self.initial_portfolio_value = 1.0
        self.current_step = self.lookback

        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.portfolio_value = self.initial_portfolio_value
        self.peak_portfolio_value = self.initial_portfolio_value
        self.current_mdd = 0.0

        # returns window + current weights + RSI + MACD
        # MACD_signal 추가
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

    def _normalize_action(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)

        if action.sum() < 1e-8:
            return np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        return action / action.sum()

    def _get_observation(self):
        # 관측값은 정규화된 features_df만 사용
        return_cols = [f"{asset}_return" for asset in self.asset_names]
        rsi_cols = [f"{asset}_RSI" for asset in self.asset_names]
        # macd_cols = [f"{asset}_MACD" for asset in self.asset_names]
        macd_signal_cols = [f"{asset}_MACD_signal" for asset in self.asset_names]

        returns_window = self.features_df[return_cols].iloc[
            self.current_step - self.lookback:self.current_step
        ].values

        rsi = self.features_df[rsi_cols].iloc[self.current_step].values
        # macd = self.features_df[macd_cols].iloc[self.current_step].values
        macd_signal = self.features_df[macd_signal_cols].iloc[self.current_step].values

        obs = np.concatenate(
            [
                returns_window.flatten(),
                self.weights,
                rsi,
                # macd,
                macd_signal,
            ]
        ).astype(np.float32)

        return obs

    def _calculate_reward(self, net_return, weights):
        if self.reward_type == "return":
            return net_return

        if self.reward_type == "sharpe":
            start = max(0, self.current_step - self.volatility_window)
            recent_returns = self.returns_df.iloc[start:self.current_step].values

            portfolio_returns = recent_returns @ weights
            volatility = float(np.std(portfolio_returns))

            if volatility < 1e-8:
                return 0.0

            return net_return / volatility

        if self.reward_type == "mdd":
            return net_return - self.lambda_mdd * self.current_mdd

        raise ValueError(f"Unsupported reward_type: {self.reward_type}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.lookback
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        self.portfolio_value = self.initial_portfolio_value
        self.peak_portfolio_value = self.initial_portfolio_value
        self.current_mdd = 0.0

        observation = self._get_observation()

        info = {
            "portfolio_value": self.portfolio_value,
            "mdd": self.current_mdd,
            "weights": self.weights.copy(),
            "step": self.current_step,
            "reward_type": self.reward_type,
        }

        return observation, info

    def step(self, action):
        new_weights = self._normalize_action(action)

        # 실제 수익률 계산은 raw 로그수익률 returns_df 사용
        asset_returns = self.returns_df.iloc[self.current_step].values.astype(np.float32)

        gross_return = float(np.dot(new_weights, asset_returns))

        turnover = float(np.sum(np.abs(new_weights - self.weights)))
        transaction_cost = turnover * self.total_cost_rate

        net_return = gross_return - transaction_cost

        self.portfolio_value *= 1.0 + net_return
        self.peak_portfolio_value = max(
            self.peak_portfolio_value,
            self.portfolio_value,
        )

        self.current_mdd = (
            self.peak_portfolio_value - self.portfolio_value
        ) / self.peak_portfolio_value

        reward = float(self._calculate_reward(net_return, new_weights))

        self.weights = new_weights
        self.current_step += 1

        reached_end = self.current_step >= len(self.features_df)
        safe_guard_triggered = self.current_mdd > self.safe_guard_mdd

        terminated = reached_end or safe_guard_triggered
        truncated = False

        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._get_observation()

        info = {
            "weights": self.weights.copy(),
            "step": self.current_step,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "reward": reward,
            "reward_type": self.reward_type,
            "portfolio_value": self.portfolio_value,
            "mdd": self.current_mdd,
            "safe_guard_triggered": safe_guard_triggered,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        print(
            f"step={self.current_step}, "
            f"value={self.portfolio_value:.4f}, "
            f"mdd={self.current_mdd:.4f}, "
            f"reward_type={self.reward_type}, "
            f"weights={np.round(self.weights, 3)}"
        )