"""
Unit tests for evaluation/baselines.py
"""

import pytest
import numpy as np
from evaluation.baselines import (
    EqualWeightStrategy, BuyAndHoldStrategy, RandomStrategy, MomentumStrategy,
    MinimumVarianceStrategy, InverseVolatilityStrategy, BaselineStrategyRegistry
)


class TestEqualWeightStrategy:
    """Tests for EqualWeightStrategy."""

    def test_get_action_returns_equal_weights(self):
        """Test that equal weights are returned."""
        strategy = EqualWeightStrategy()

        class MockEnv:
            n_assets = 3

        env = MockEnv()
        action = strategy.get_action(env, step=0)

        # Action includes assets + cash dimension
        assert len(action) == env.n_assets + 1
        # First n_assets should be equal (1.0), last is cash weight
        assert np.allclose(action[:-1], np.ones(env.n_assets))
        assert action[-1] == strategy.cash_weight

    def test_name(self):
        """Test strategy name."""
        strategy = EqualWeightStrategy()
        assert strategy.name == 'equal_weight'


class TestBuyAndHoldStrategy:
    """Tests for BuyAndHoldStrategy."""

    def test_first_step_returns_equal_weights(self):
        """Test that first step returns equal weights."""
        strategy = BuyAndHoldStrategy()

        class MockEnv:
            n_assets = 3
            weights = np.array([0.0, 0.0, 0.0])

        env = MockEnv()
        action = strategy.get_action(env, step=0)

        # Action includes assets + cash dimension
        assert len(action) == env.n_assets + 1
        assert np.allclose(action[:-1], np.ones(env.n_assets))
        assert action[-1] == strategy.cash_weight

    def test_subsequent_steps_maintain_weights(self):
        """Test that subsequent steps maintain weights."""
        strategy = BuyAndHoldStrategy()

        class MockEnv:
            n_assets = 3
            weights = np.array([0.4, 0.35, 0.25])
            cash = 100.0
            portfolio_value = 10000.0

        env = MockEnv()

        # First step
        strategy.get_action(env, step=0)

        # Second step - should maintain weights including cash
        action = strategy.get_action(env, step=1)

        # Expected: asset weights in log space + cash ratio in log space
        expected_assets = np.log(env.weights + 1e-8)
        expected_cash = np.log(env.cash / env.portfolio_value + 1e-8)
        expected = np.append(expected_assets, expected_cash)

        assert np.allclose(action, expected)

    def test_reset_resets_first_step_flag(self):
        """Test that reset resets the first step flag."""
        strategy = BuyAndHoldStrategy()

        class MockEnv:
            n_assets = 3
            weights = np.array([0.4, 0.35, 0.25])
            cash = 100.0
            portfolio_value = 10000.0

        env = MockEnv()

        # First episode
        strategy.get_action(env, step=0)
        strategy.get_action(env, step=1)

        # Reset
        strategy.reset()

        # Should return equal weights again (including cash dimension)
        action = strategy.get_action(env, step=0)
        assert len(action) == env.n_assets + 1
        assert np.allclose(action[:-1], np.ones(env.n_assets))
        assert action[-1] == strategy.cash_weight


class TestRandomStrategy:
    """Tests for RandomStrategy."""

    def test_get_action_returns_random_weights(self):
        """Test that random weights are returned."""
        strategy = RandomStrategy(seed=42)

        class MockEnv:
            n_assets = 3

        env = MockEnv()
        action1 = strategy.get_action(env, step=0)
        action2 = strategy.get_action(env, step=1)

        # Action includes assets + cash dimension
        assert len(action1) == env.n_assets + 1
        assert len(action2) == env.n_assets + 1
        assert not np.allclose(action1, action2)

    def test_seed_produces_reproducible_results(self):
        """Test that seed produces reproducible results."""
        strategy1 = RandomStrategy(seed=42)
        strategy2 = RandomStrategy(seed=42)

        class MockEnv:
            n_assets = 3

        env = MockEnv()

        action1 = strategy1.get_action(env, step=0)
        action2 = strategy2.get_action(env, step=0)

        assert np.allclose(action1, action2)


class TestMomentumStrategy:
    """Tests for MomentumStrategy."""

    @pytest.fixture
    def mock_env_with_prices(self):
        """Create mock environment with price history."""
        class MockEnv:
            n_assets = 3
            tickers = ['A', 'B', 'C']

            def __init__(self):
                self.price_history = {}
                for i in range(50):
                    self.price_history[i] = np.array([100 + i, 100 - i*0.5, 100 + i*0.3])

            def _get_prices(self, step):
                return self.price_history.get(step, np.array([100, 100, 100]))

        return MockEnv()

    def test_get_action_with_insufficient_history(self, mock_env_with_prices):
        """Test that equal weights are returned with insufficient history."""
        strategy = MomentumStrategy(lookback=20)
        action = strategy.get_action(mock_env_with_prices, step=10)

        # Action includes assets + cash dimension
        assert len(action) == mock_env_with_prices.n_assets + 1
        assert np.allclose(action[:-1], np.ones(mock_env_with_prices.n_assets))
        assert action[-1] == strategy.cash_weight

    def test_get_action_with_sufficient_history(self, mock_env_with_prices):
        """Test that momentum-based weights are returned."""
        strategy = MomentumStrategy(lookback=20)
        action = strategy.get_action(mock_env_with_prices, step=30)

        # Action includes assets + cash dimension
        assert len(action) == mock_env_with_prices.n_assets + 1
        # Asset weights should be positive (cash weight is negative by default)
        assert (action[:-1] > 0).all()

    def test_name(self):
        """Test strategy name."""
        strategy = MomentumStrategy(lookback=20)
        assert strategy.name == 'momentum_20'


class TestMinimumVarianceStrategy:
    """Tests for MinimumVarianceStrategy."""

    @pytest.fixture
    def mock_env_with_returns(self):
        """Create mock environment with return history."""
        class MockEnv:
            n_assets = 3
            tickers = ['A', 'B', 'C']

            def __init__(self):
                np.random.seed(42)
                self.returns_history = {}
                base_prices = np.array([100.0, 100.0, 100.0])
                for i in range(100):
                    returns = np.random.normal(0.001, 0.02, 3)
                    base_prices = base_prices * (1 + returns)
                    self.returns_history[i] = base_prices.copy()

            def _get_prices(self, step):
                return self.returns_history.get(step, np.array([100, 100, 100]))

        return MockEnv()

    def test_get_action_with_insufficient_history(self, mock_env_with_returns):
        """Test that equal weights are returned with insufficient history."""
        strategy = MinimumVarianceStrategy(lookback=60)
        action = strategy.get_action(mock_env_with_returns, step=30)

        # Action includes assets + cash dimension
        assert len(action) == mock_env_with_returns.n_assets + 1
        assert np.allclose(action[:-1], np.ones(mock_env_with_returns.n_assets))
        assert action[-1] == strategy.cash_weight

    def test_get_action_with_sufficient_history(self, mock_env_with_returns):
        """Test that minimum variance weights are computed."""
        strategy = MinimumVarianceStrategy(lookback=60)
        action = strategy.get_action(mock_env_with_returns, step=70)

        # Action includes assets + cash dimension
        assert len(action) == mock_env_with_returns.n_assets + 1

    def test_name(self):
        """Test strategy name."""
        strategy = MinimumVarianceStrategy(lookback=60)
        assert strategy.name == 'min_variance_60'


class TestInverseVolatilityStrategy:
    """Tests for InverseVolatilityStrategy."""

    @pytest.fixture
    def mock_env_with_volatility(self):
        """Create mock environment with price history."""
        class MockEnv:
            n_assets = 3
            tickers = ['A', 'B', 'C']

            def __init__(self):
                np.random.seed(42)
                self.price_history = {}
                base_prices = np.array([100.0, 100.0, 100.0])
                for i in range(100):
                    returns = np.random.normal([0.001, 0.001, 0.001], [0.01, 0.02, 0.03])
                    base_prices = base_prices * (1 + returns)
                    self.price_history[i] = base_prices.copy()

            def _get_prices(self, step):
                return self.price_history.get(step, np.array([100, 100, 100]))

        return MockEnv()

    def test_get_action_with_insufficient_history(self, mock_env_with_volatility):
        """Test that equal weights are returned with insufficient history."""
        strategy = InverseVolatilityStrategy(lookback=30)
        action = strategy.get_action(mock_env_with_volatility, step=20)

        # Action includes assets + cash dimension
        assert len(action) == mock_env_with_volatility.n_assets + 1
        assert np.allclose(action[:-1], np.ones(mock_env_with_volatility.n_assets))
        assert action[-1] == strategy.cash_weight

    def test_get_action_with_sufficient_history(self, mock_env_with_volatility):
        """Test that inverse volatility weights are computed."""
        strategy = InverseVolatilityStrategy(lookback=30)
        action = strategy.get_action(mock_env_with_volatility, step=50)

        # Action includes assets + cash dimension
        assert len(action) == mock_env_with_volatility.n_assets + 1
        # Asset weights should be positive (inverse of volatility)
        assert (action[:-1] > 0).all()

    def test_name(self):
        """Test strategy name."""
        strategy = InverseVolatilityStrategy(lookback=30)
        assert strategy.name == 'inverse_vol_30'


class TestBaselineStrategyRegistry:
    """Tests for BaselineStrategyRegistry."""

    def test_init_registers_default_strategies(self):
        """Test that default strategies are registered."""
        registry = BaselineStrategyRegistry()
        strategies = registry.list_strategies()

        assert 'equal_weight' in strategies
        assert 'buy_and_hold' in strategies
        assert 'random' in strategies
        assert 'momentum_20' in strategies
        assert 'min_variance_60' in strategies
        assert 'inverse_vol_30' in strategies

    def test_get_strategy(self):
        """Test getting a registered strategy."""
        registry = BaselineStrategyRegistry()
        strategy = registry.get('equal_weight')

        assert isinstance(strategy, EqualWeightStrategy)

    def test_get_unknown_strategy_raises_error(self):
        """Test that getting unknown strategy raises error."""
        registry = BaselineStrategyRegistry()

        with pytest.raises(ValueError, match="Unknown strategy"):
            registry.get('unknown_strategy')

    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""
        class CustomStrategy(EqualWeightStrategy):
            def __init__(self):
                super().__init__()
                self.name = 'custom'

        registry = BaselineStrategyRegistry()
        custom = CustomStrategy()
        registry.register(custom)

        assert 'custom' in registry.list_strategies()
        retrieved = registry.get('custom')
        assert isinstance(retrieved, CustomStrategy)

    def test_list_strategies(self):
        """Test listing all strategies."""
        registry = BaselineStrategyRegistry()
        strategies = registry.list_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) >= 6
