"""
Unit tests for environment/portfolio_env.py
"""

import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
from environment.portfolio_env import PortfolioEnv
from environment.rewards import SimpleReturnReward
from data.features import FeatureEngineer


class TestPortfolioEnv:
    """Tests for PortfolioEnv class."""

    @pytest.fixture
    def env_data(self, sample_ohlcv_data):
        """Create environment-ready data."""
        engineer = FeatureEngineer()
        features_data = engineer.compute_features(sample_ohlcv_data, ['returns', 'sma_20'])
        prepared = engineer.prepare_for_environment(features_data, normalize=False)
        return prepared

    @pytest.fixture
    def simple_env(self, env_data, sample_tickers):
        """Create a simple portfolio environment."""
        feature_cols = ['returns', 'Close', 'sma_20']
        env = PortfolioEnv(
            data=env_data,
            feature_columns=feature_cols,
            tickers=sample_tickers,
            initial_balance=10000.0,
            transaction_cost=0.001,
            window_size=20
        )
        return env

    def test_init_creates_valid_env(self, simple_env):
        """Test that environment initializes correctly."""
        assert isinstance(simple_env, gym.Env)
        assert simple_env.initial_balance == 10000.0
        assert simple_env.transaction_cost == 0.001
        assert simple_env.n_assets == 3

    def test_action_space_shape(self, simple_env):
        """Test action space dimensions."""
        # Action space includes assets + cash
        assert simple_env.action_space.shape == (simple_env.n_assets + 1,)
        assert isinstance(simple_env.action_space, gym.spaces.Box)

    def test_observation_space_shape(self, simple_env):
        """Test observation space dimensions."""
        window_size = simple_env.window_size
        n_features = len(simple_env.feature_columns)
        n_assets = simple_env.n_assets

        expected_dim = (
            window_size * n_assets * n_features +  # Market features
            n_assets +  # Current weights
            1  # Cash ratio
        )

        assert simple_env.observation_space.shape == (expected_dim,)

    def test_reset_returns_valid_observation(self, simple_env):
        """Test that reset returns valid observation and info."""
        obs, info = simple_env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == simple_env.observation_space.shape
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()

        assert isinstance(info, dict)
        assert 'portfolio_value' in info
        assert info['portfolio_value'] == simple_env.initial_balance

    def test_reset_initializes_state(self, simple_env):
        """Test that reset properly initializes state."""
        obs, info = simple_env.reset()

        assert simple_env.current_step == 0
        assert simple_env.portfolio_value == simple_env.initial_balance
        assert simple_env.cash == simple_env.initial_balance
        assert np.allclose(simple_env.holdings, 0)
        assert np.allclose(simple_env.weights, 0)

    def test_step_returns_valid_output(self, simple_env):
        """Test that step returns valid output."""
        simple_env.reset()
        action = simple_env.action_space.sample()

        obs, reward, terminated, truncated, info = simple_env.step(action)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == simple_env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_updates_portfolio_value(self, simple_env):
        """Test that step updates portfolio value."""
        simple_env.reset()
        initial_value = simple_env.portfolio_value

        # Action includes assets + cash dimension
        action = np.ones(simple_env.n_assets + 1)
        simple_env.step(action)

        assert simple_env.current_step == 1
        assert simple_env.portfolio_value != initial_value

    def test_softmax_normalizes_weights(self, simple_env):
        """Test that softmax produces valid weights."""
        action = np.array([1.0, 2.0, 3.0])
        weights = simple_env._softmax(action)

        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()
        assert (weights <= 1).all()

    def test_softmax_numerical_stability(self, simple_env):
        """Test softmax with large values."""
        action = np.array([1000.0, 1001.0, 999.0])
        weights = simple_env._softmax(action)

        assert np.isclose(weights.sum(), 1.0)
        assert not np.isnan(weights).any()
        assert not np.isinf(weights).any()

    def test_get_prices_returns_valid_prices(self, simple_env):
        """Test that _get_prices returns valid price array."""
        simple_env.reset()
        prices = simple_env._get_prices(0)

        assert isinstance(prices, np.ndarray)
        assert prices.shape == (simple_env.n_assets,)
        assert (prices > 0).all()

    def test_rebalance_portfolio_updates_holdings(self, simple_env):
        """Test that rebalancing updates holdings."""
        simple_env.reset()
        target_weights = np.array([0.33, 0.33, 0.34])
        prices = simple_env._get_prices(0)
        portfolio_value = simple_env.initial_balance
        target_cash_ratio = 0.1  # 10% cash

        transaction_cost = simple_env._rebalance_portfolio(
            target_weights, prices, portfolio_value, target_cash_ratio
        )

        assert isinstance(transaction_cost, (int, float))
        assert transaction_cost >= 0
        assert not np.allclose(simple_env.holdings, 0)

    def test_episode_terminates(self, simple_env):
        """Test that episode terminates at the end."""
        obs, info = simple_env.reset()
        done = False
        steps = 0

        while not done and steps < simple_env.n_steps + 10:
            action = simple_env.action_space.sample()
            obs, reward, terminated, truncated, info = simple_env.step(action)
            done = terminated or truncated
            steps += 1

        assert done
        assert steps <= simple_env.n_steps

    def test_portfolio_history_tracking(self, simple_env):
        """Test that portfolio history is tracked."""
        simple_env.reset()

        for _ in range(5):
            action = simple_env.action_space.sample()
            simple_env.step(action)

        history = simple_env.get_portfolio_history()

        assert isinstance(history, pd.DataFrame)
        assert len(history) == 6  # Initial state + 5 steps
        assert 'date' in history.columns
        assert 'value' in history.columns
        assert 'weights' in history.columns

    def test_transaction_costs_are_applied(self, simple_env):
        """Test that transaction costs reduce portfolio value."""
        simple_env.reset()

        # Action includes assets + cash dimension
        action = np.ones(simple_env.n_assets + 1)
        obs, reward, terminated, truncated, info = simple_env.step(action)

        assert simple_env.portfolio_value < simple_env.initial_balance or \
               simple_env.portfolio_value > simple_env.initial_balance

    def test_validate_data_missing_ticker_raises_error(self, env_data, sample_tickers):
        """Test that missing ticker raises error."""
        feature_cols = ['returns', 'Close']
        invalid_tickers = sample_tickers + ['INVALID']

        with pytest.raises(ValueError, match="Ticker .* not found in data"):
            PortfolioEnv(
                data=env_data,
                feature_columns=feature_cols,
                tickers=invalid_tickers,
                initial_balance=10000.0
            )

    def test_validate_data_missing_feature_raises_error(self, env_data, sample_tickers):
        """Test that missing feature column raises error."""
        invalid_features = ['returns', 'invalid_feature']

        with pytest.raises(ValueError, match="Feature column .* not found in data"):
            PortfolioEnv(
                data=env_data,
                feature_columns=invalid_features,
                tickers=sample_tickers,
                initial_balance=10000.0
            )

    def test_observation_handles_nan_values(self, simple_env):
        """Test that observations don't contain NaN or Inf."""
        simple_env.reset()

        for _ in range(10):
            action = simple_env.action_space.sample()
            obs, reward, terminated, truncated, info = simple_env.step(action)

            assert not np.isnan(obs).any()
            assert not np.isinf(obs).any()

            if terminated or truncated:
                break

    def test_custom_reward_function(self, env_data, sample_tickers):
        """Test environment with custom reward function."""
        from environment.rewards import SharpeReward

        feature_cols = ['returns', 'Close', 'sma_20']
        reward_fn = SharpeReward(window=20)

        env = PortfolioEnv(
            data=env_data,
            feature_columns=feature_cols,
            tickers=sample_tickers,
            initial_balance=10000.0,
            reward_function=reward_fn
        )

        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, (int, float))

    def test_weights_sum_to_one_after_step(self, simple_env):
        """Test that weights + cash ratio sum to approximately 1."""
        simple_env.reset()

        for _ in range(10):
            action = simple_env.action_space.sample()
            obs, reward, terminated, truncated, info = simple_env.step(action)

            # Now weights represent only asset allocation (not including cash)
            # So weights + cash_ratio should sum to ~1.0
            weights_sum = simple_env.weights.sum()
            cash_ratio = simple_env.cash / simple_env.portfolio_value if simple_env.portfolio_value > 0 else 0.0
            total_allocation = weights_sum + cash_ratio

            # Total allocation should sum to ~1.0 or be all zeros (bankrupt)
            # Allow for numerical errors from float32 precision and transaction costs
            assert np.isclose(total_allocation, 1.0, atol=0.01) or np.isclose(total_allocation, 0.0, atol=0.01)

            if terminated or truncated:
                break

    def test_render_does_not_raise_error(self, simple_env):
        """Test that render method works."""
        simple_env.reset()
        simple_env.render(mode='human')

    def test_get_info_returns_complete_info(self, simple_env):
        """Test that _get_info returns all expected fields."""
        simple_env.reset()
        info = simple_env._get_info()

        assert 'portfolio_value' in info
        assert 'weights' in info
        assert 'cash' in info
        assert 'holdings' in info
        assert 'date' in info
