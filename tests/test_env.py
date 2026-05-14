import numpy as np
import pandas as pd
import pytest

from src.rl.env import PortfolioEnv


@pytest.fixture
def sample_data():
    tickers = ["SPY", "QQQ", "IWM"]
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    returns_df = pd.DataFrame(
        np.random.normal(0, 0.01, size=(100, len(tickers))),
        index=dates,
        columns=tickers,
    )

    feature_data = {}
    for ticker in tickers:
        feature_data[f"{ticker}_return"] = np.random.normal(0, 1, size=100)
        feature_data[f"{ticker}_RSI"] = np.random.normal(0, 1, size=100)
        feature_data[f"{ticker}_MACD_signal"] = np.random.normal(0, 1, size=100)

    features_df = pd.DataFrame(feature_data, index=dates)

    return returns_df, features_df, tickers


def test_env_reset_observation_shape(sample_data):
    returns_df, features_df, tickers = sample_data
    lookback = 30

    env = PortfolioEnv(returns_df, features_df, lookback=lookback)

    obs, info = env.reset()

    expected_obs_dim = (lookback + 3) * len(tickers) + 3

    assert obs.shape == (expected_obs_dim,)
    assert obs.dtype == np.float32
    assert env.observation_space.shape == (expected_obs_dim,)
    assert info["step"] == lookback
    assert np.isclose(info["weights"].sum(), 1.0)


def test_env_step_returns_valid_values(sample_data):
    returns_df, features_df, _ = sample_data

    env = PortfolioEnv(returns_df, features_df, lookback=30)
    env.reset()

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert isinstance(terminated, bool)
    assert truncated is False

    assert "weights" in info
    assert "gross_return" in info
    assert "transaction_cost" in info
    assert "net_return" in info
    assert "reward" in info
    assert "reward_type" in info
    assert "portfolio_value" in info
    assert "mdd" in info
    assert "safe_guard_triggered" in info


def test_weights_sum_to_one_after_step(sample_data):
    returns_df, features_df, _ = sample_data

    env = PortfolioEnv(returns_df, features_df, lookback=30)
    env.reset()

    action = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    _, _, _, _, info = env.step(action)

    assert np.isclose(info["weights"].sum(), 1.0)


def test_zero_action_converts_to_equal_weights(sample_data):
    returns_df, features_df, tickers = sample_data

    env = PortfolioEnv(returns_df, features_df, lookback=30)
    env.reset()

    zero_action = np.zeros(len(tickers), dtype=np.float32)
    _, _, _, _, info = env.step(zero_action)

    expected_weights = np.ones(len(tickers), dtype=np.float32) / len(tickers)

    assert np.allclose(info["weights"], expected_weights)


def test_transaction_cost_is_applied(sample_data):
    returns_df, features_df, _ = sample_data

    env = PortfolioEnv(
    returns_df,
    features_df,
    lookback=30,
    reward_type="return",
    )
    env.reset()

    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    _, reward, _, _, info = env.step(action)

    assert info["transaction_cost"] > 0
    assert np.isclose(reward, info["net_return"])
    assert info["reward_type"] == "return"


def test_safe_guard_triggers_when_mdd_exceeds_limit(sample_data):
    returns_df, features_df, tickers = sample_data

    returns_df.iloc[:] = -0.2

    env = PortfolioEnv(returns_df, features_df, lookback=30)
    env.reset()

    action = np.ones(len(tickers), dtype=np.float32) / len(tickers)

    _, _, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert info["safe_guard_triggered"] is True
    assert info["mdd"] > 0.15


def test_env_runs_multiple_random_steps(sample_data):
    returns_df, features_df, _ = sample_data

    env = PortfolioEnv(returns_df, features_df, lookback=30)
    obs, _ = env.reset()

    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        assert obs.shape == env.observation_space.shape
        assert not np.isnan(reward)
        assert truncated is False

        if terminated:
            break


def test_reward_type_return_uses_net_return(sample_data):
    returns_df, features_df, _ = sample_data

    env = PortfolioEnv(
        returns_df,
        features_df,
        lookback=30,
        reward_type="return",
    )
    env.reset()

    action = env.action_space.sample()
    _, reward, _, _, info = env.step(action)

    assert np.isclose(reward, info["net_return"])
    assert not np.isnan(reward)
    assert info["reward_type"] == "return"


def test_reward_type_sharpe_returns_valid_reward(sample_data):
    returns_df, features_df, _ = sample_data

    env = PortfolioEnv(
        returns_df,
        features_df,
        lookback=30,
        reward_type="sharpe",
        volatility_window=20,
    )
    env.reset()

    action = env.action_space.sample()
    _, reward, _, _, info = env.step(action)

    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert not np.isinf(reward)
    assert info["reward_type"] == "sharpe"


def test_reward_type_mdd_applies_lambda_penalty(sample_data):
    returns_df, features_df, tickers = sample_data

    returns_df.iloc[:] = -0.05
    lambda_mdd = 2.0

    env = PortfolioEnv(
        returns_df,
        features_df,
        lookback=30,
        reward_type="mdd",
        lambda_mdd=lambda_mdd,
    )
    env.reset()

    action = np.ones(len(tickers), dtype=np.float32) / len(tickers)
    _, reward, _, _, info = env.step(action)

    expected_reward = info["net_return"] - lambda_mdd * info["mdd"]

    assert np.isclose(reward, expected_reward)
    assert not np.isnan(reward)
    assert info["reward_type"] == "mdd"


def test_risk_vector_obs_shape_with_default(sample_data):
    """risk_vector 미전달 시 obs shape이 기존 대비 +3임을 검증합니다."""
    returns_df, features_df, tickers = sample_data
    lookback = 30

    env = PortfolioEnv(returns_df, features_df, lookback=lookback)
    obs, _ = env.reset()

    expected_obs_dim = (lookback + 3) * len(tickers) + 3

    assert obs.shape == (expected_obs_dim,)
    assert env.observation_space.shape == (expected_obs_dim,)


def test_risk_vector_default_is_zeros(sample_data):
    """기본 risk_vector가 np.zeros(3)인지 검증합니다."""
    returns_df, features_df, _ = sample_data

    env = PortfolioEnv(returns_df, features_df, lookback=30)

    assert env.risk_vector.shape == (3,)
    assert np.all(env.risk_vector == 0.0)
    assert env.risk_vector.dtype == np.float32


def test_risk_vector_appended_to_obs(sample_data):
    """risk_vector가 obs 마지막 3개 원소에 반영되는지 검증합니다."""
    returns_df, features_df, _ = sample_data
    lookback = 30
    risk = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    env = PortfolioEnv(returns_df, features_df, lookback=lookback, risk_vector=risk)
    obs, _ = env.reset()

    assert np.allclose(obs[-3:], risk)


def test_set_risk_vector_updates_obs(sample_data):
    """set_risk_vector 호출 후 obs 마지막 3개 원소가 갱신되는지 검증합니다."""
    returns_df, features_df, _ = sample_data

    env = PortfolioEnv(returns_df, features_df, lookback=30)
    env.reset()

    new_risk = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    env.set_risk_vector(new_risk)

    obs = env._get_observation()

    assert np.allclose(obs[-3:], new_risk)


def test_risk_vector_wrong_shape_raises_error(sample_data):
    """risk_vector shape이 (3,)이 아니면 ValueError를 발생시키는지 검증합니다."""
    returns_df, features_df, _ = sample_data

    with pytest.raises(ValueError, match="risk_vector must have shape"):
        PortfolioEnv(returns_df, features_df, lookback=30, risk_vector=[1.0, 0.0])


def test_set_risk_vector_wrong_shape_raises_error(sample_data):
    """set_risk_vector에 잘못된 shape 전달 시 ValueError를 발생시키는지 검증합니다."""
    returns_df, features_df, _ = sample_data

    env = PortfolioEnv(returns_df, features_df, lookback=30)

    with pytest.raises(ValueError, match="risk_vector must have shape"):
        env.set_risk_vector(np.array([1.0, 0.0]))


def test_terminated_is_python_bool(sample_data):
    """step()의 terminated가 Python bool 타입인지 검증합니다."""
    returns_df, features_df, _ = sample_data

    env = PortfolioEnv(returns_df, features_df, lookback=30)
    env.reset()

    action = env.action_space.sample()
    _, _, terminated, truncated, _ = env.step(action)

    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_invalid_reward_type_raises_error(sample_data):
    returns_df, features_df, _ = sample_data

    with pytest.raises(ValueError):
        PortfolioEnv(
            returns_df,
            features_df,
            lookback=30,
            reward_type="invalid",
        )


def test_observation_uses_only_spec_features(sample_data):
    returns_df, features_df, tickers = sample_data
    lookback = 30

    env = PortfolioEnv(returns_df, features_df, lookback=lookback)
    obs, _ = env.reset()

    expected_obs_dim = (lookback + 3) * len(tickers) + 3

    assert obs.shape == (expected_obs_dim,)
    assert env.observation_space.shape == (expected_obs_dim,)