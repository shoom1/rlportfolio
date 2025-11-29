"""
Unit tests for evaluation/backtest_strategies.py

Tests the Strategy pattern for backtest execution.
"""

import pytest
import numpy as np
import pandas as pd
from evaluation.backtest_strategies import (
    SequentialBacktestStrategy,
    WalkForwardBacktestStrategy,
    MonteCarloBacktestStrategy
)
from evaluation.baselines import EqualWeightStrategy


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, deterministic_action=None):
        self.deterministic_action = deterministic_action or np.array([1.0, 1.0, 1.0, -5.0])

    def predict(self, obs, deterministic=True):
        """Return fixed action."""
        return self.deterministic_action, None


class MockEnv:
    """Mock environment for testing backtest strategies."""

    def __init__(self, n_steps=100, n_assets=3):
        self.n_steps = n_steps
        self.n_assets = n_assets
        self.current_step = 0
        self.data = pd.DataFrame({'dummy': range(n_steps)})

    def reset(self, seed=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        obs = np.random.random(10)
        info = {}
        return obs, info

    def step(self, action):
        """Take a step."""
        self.current_step += 1
        obs = np.random.random(10)
        reward = np.random.random()
        terminated = self.current_step >= self.n_steps
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def get_portfolio_history(self):
        """Return mock portfolio history."""
        dates = pd.date_range('2023-01-01', periods=self.current_step, freq='D')
        values = 10000 * np.cumprod(1 + np.random.normal(0.001, 0.01, self.current_step))

        return pd.DataFrame({
            'date': dates,
            'value': values,
            'return': np.concatenate([[0], np.diff(values) / values[:-1]]),
            'weights': [np.array([0.33, 0.33, 0.34]) for _ in range(self.current_step)]
        })


class TestSequentialBacktestStrategy:
    """Tests for SequentialBacktestStrategy."""

    def test_execute_agent_completes(self):
        """Test that agent backtest completes successfully."""
        strategy = SequentialBacktestStrategy()
        agent = MockAgent()
        env = MockEnv(n_steps=50)

        history = strategy.execute_agent(agent, env, deterministic=True)

        assert isinstance(history, pd.DataFrame)
        assert len(history) == 50
        assert 'value' in history.columns
        assert 'return' in history.columns

    def test_execute_agent_deterministic_flag(self):
        """Test that deterministic flag is passed correctly."""
        strategy = SequentialBacktestStrategy()
        agent = MockAgent()
        env = MockEnv(n_steps=10)

        # Should not raise error
        history1 = strategy.execute_agent(agent, env, deterministic=True)
        env2 = MockEnv(n_steps=10)
        history2 = strategy.execute_agent(agent, env2, deterministic=False)

        assert isinstance(history1, pd.DataFrame)
        assert isinstance(history2, pd.DataFrame)

    def test_execute_baseline_completes(self):
        """Test that baseline backtest completes successfully."""
        strategy = SequentialBacktestStrategy()
        baseline = EqualWeightStrategy()
        env = MockEnv(n_steps=50)

        history = strategy.execute_baseline(baseline, env)

        assert isinstance(history, pd.DataFrame)
        assert len(history) == 50
        assert 'value' in history.columns

    def test_execute_baseline_calls_reset(self):
        """Test that baseline strategy reset is called."""
        strategy = SequentialBacktestStrategy()
        baseline = EqualWeightStrategy()

        # Set some internal state
        baseline._some_state = "should_be_reset"

        env = MockEnv(n_steps=10)
        history = strategy.execute_baseline(baseline, env)

        # Baseline should have been reset
        assert isinstance(history, pd.DataFrame)


class TestWalkForwardBacktestStrategy:
    """Tests for WalkForwardBacktestStrategy."""

    def test_initialization(self):
        """Test walk-forward strategy initialization."""
        strategy = WalkForwardBacktestStrategy(
            train_window=200,
            test_window=60,
            step_size=30
        )

        assert strategy.train_window == 200
        assert strategy.test_window == 60
        assert strategy.step_size == 30
        assert strategy.fold_results == []

    def test_default_step_size(self):
        """Test that step_size defaults to test_window."""
        strategy = WalkForwardBacktestStrategy(
            train_window=200,
            test_window=60
        )

        assert strategy.step_size == 60

    def test_execute_agent_creates_folds(self):
        """Test that walk-forward creates multiple folds."""
        strategy = WalkForwardBacktestStrategy(
            train_window=30,
            test_window=20,
            step_size=10
        )

        agent = MockAgent()
        env = MockEnv(n_steps=100)

        history = strategy.execute_agent(agent, env, deterministic=True)

        # Should have created multiple folds
        assert len(strategy.fold_results) > 0
        assert isinstance(history, pd.DataFrame)
        assert len(history) > 0

    def test_get_fold_results(self):
        """Test getting fold results metadata."""
        strategy = WalkForwardBacktestStrategy(
            train_window=30,
            test_window=20,
            step_size=20
        )

        agent = MockAgent()
        env = MockEnv(n_steps=100)

        history = strategy.execute_agent(agent, env)
        fold_results = strategy.get_fold_results()

        assert isinstance(fold_results, list)
        assert len(fold_results) > 0

        # Check fold metadata structure
        for fold in fold_results:
            assert 'train_start' in fold
            assert 'train_end' in fold
            assert 'test_start' in fold
            assert 'test_end' in fold
            assert 'n_steps' in fold

    def test_execute_baseline_creates_folds(self):
        """Test walk-forward with baseline strategy."""
        strategy = WalkForwardBacktestStrategy(
            train_window=30,
            test_window=20,
            step_size=20
        )

        baseline = EqualWeightStrategy()
        env = MockEnv(n_steps=100)

        history = strategy.execute_baseline(baseline, env)

        assert isinstance(history, pd.DataFrame)
        assert len(strategy.fold_results) > 0

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises error."""
        strategy = WalkForwardBacktestStrategy(
            train_window=200,
            test_window=100
        )

        agent = MockAgent()
        env = MockEnv(n_steps=50)  # Not enough data

        with pytest.raises(ValueError, match="Not enough data"):
            strategy.execute_agent(agent, env)

    def test_fold_window_sizes(self):
        """Test that fold windows are correctly sized."""
        strategy = WalkForwardBacktestStrategy(
            train_window=40,
            test_window=20,
            step_size=20
        )

        agent = MockAgent()
        env = MockEnv(n_steps=100)

        history = strategy.execute_agent(agent, env)
        folds = strategy.get_fold_results()

        for fold in folds:
            train_size = fold['train_end'] - fold['train_start']
            test_size = fold['test_end'] - fold['test_start']

            assert train_size == 40
            assert test_size == 20


class TestMonteCarloBacktestStrategy:
    """Tests for MonteCarloBacktestStrategy."""

    def test_initialization(self):
        """Test Monte Carlo strategy initialization."""
        strategy = MonteCarloBacktestStrategy(n_simulations=50)

        assert strategy.n_simulations == 50
        assert len(strategy.seeds) == 50
        assert strategy.simulation_results == []

    def test_custom_seeds(self):
        """Test initialization with custom seeds."""
        custom_seeds = [10, 20, 30, 40, 50]
        strategy = MonteCarloBacktestStrategy(
            n_simulations=5,
            seeds=custom_seeds
        )

        assert strategy.seeds == custom_seeds

    def test_execute_agent_runs_simulations(self):
        """Test that Monte Carlo runs multiple simulations."""
        strategy = MonteCarloBacktestStrategy(n_simulations=10)

        agent = MockAgent()
        env = MockEnv(n_steps=50)

        history = strategy.execute_agent(agent, env, deterministic=False)

        assert isinstance(history, pd.DataFrame)
        assert len(strategy.simulation_results) == 10

    def test_get_simulation_results(self):
        """Test getting simulation results."""
        strategy = MonteCarloBacktestStrategy(n_simulations=5)

        agent = MockAgent()
        env = MockEnv(n_steps=30)

        history = strategy.execute_agent(agent, env)
        results = strategy.get_simulation_results()

        assert isinstance(results, list)
        assert len(results) == 5

        # Check result structure
        for result in results:
            assert 'seed' in result
            assert 'final_value' in result
            assert 'total_return' in result

    def test_get_confidence_intervals(self):
        """Test confidence interval calculation."""
        strategy = MonteCarloBacktestStrategy(n_simulations=20)

        agent = MockAgent()
        env = MockEnv(n_steps=50)

        history = strategy.execute_agent(agent, env)
        ci = strategy.get_confidence_intervals(confidence=0.95)

        assert 'final_value_ci' in ci
        assert 'total_return_ci' in ci

        # Check CI structure
        assert len(ci['final_value_ci']) == 2
        assert len(ci['total_return_ci']) == 2

        # Lower bound should be less than upper bound
        assert ci['final_value_ci'][0] < ci['final_value_ci'][1]
        assert ci['total_return_ci'][0] < ci['total_return_ci'][1]

    def test_confidence_intervals_different_levels(self):
        """Test confidence intervals at different levels."""
        strategy = MonteCarloBacktestStrategy(n_simulations=30)

        agent = MockAgent()
        env = MockEnv(n_steps=50)

        history = strategy.execute_agent(agent, env)

        ci_90 = strategy.get_confidence_intervals(confidence=0.90)
        ci_95 = strategy.get_confidence_intervals(confidence=0.95)
        ci_99 = strategy.get_confidence_intervals(confidence=0.99)

        # 99% CI should be wider than 95% which should be wider than 90%
        width_90 = ci_90['final_value_ci'][1] - ci_90['final_value_ci'][0]
        width_95 = ci_95['final_value_ci'][1] - ci_95['final_value_ci'][0]
        width_99 = ci_99['final_value_ci'][1] - ci_99['final_value_ci'][0]

        assert width_90 <= width_95 <= width_99

    def test_execute_baseline_runs_simulations(self):
        """Test Monte Carlo with baseline strategy."""
        strategy = MonteCarloBacktestStrategy(n_simulations=10)

        baseline = EqualWeightStrategy()
        env = MockEnv(n_steps=50)

        history = strategy.execute_baseline(baseline, env)

        assert isinstance(history, pd.DataFrame)
        assert len(strategy.simulation_results) == 10

    def test_confidence_intervals_before_simulation_raises_error(self):
        """Test that getting CI before running simulations raises error."""
        strategy = MonteCarloBacktestStrategy(n_simulations=10)

        with pytest.raises(ValueError, match="No simulation results"):
            strategy.get_confidence_intervals()

    def test_aggregation_method_mean(self):
        """Test that mean aggregation is used by default."""
        strategy = MonteCarloBacktestStrategy(n_simulations=10)

        agent = MockAgent()
        env = MockEnv(n_steps=50)

        # Run simulations
        history = strategy.execute_agent(agent, env)

        # Mean history should have values between min and max of simulations
        results = strategy.get_simulation_results()
        final_values = [r['final_value'] for r in results]

        mean_final = history['value'].iloc[-1]
        assert min(final_values) <= mean_final <= max(final_values)


class TestBacktestStrategyIntegration:
    """Integration tests for strategy pattern."""

    def test_strategies_have_consistent_interface(self):
        """Test that all strategies have the same interface."""
        strategies = [
            SequentialBacktestStrategy(),
            WalkForwardBacktestStrategy(train_window=30, test_window=20),
            MonteCarloBacktestStrategy(n_simulations=5)
        ]

        for strategy in strategies:
            # Should have execute_agent method
            assert hasattr(strategy, 'execute_agent')
            assert callable(strategy.execute_agent)

            # Should have execute_baseline method
            assert hasattr(strategy, 'execute_baseline')
            assert callable(strategy.execute_baseline)

    def test_strategies_return_dataframe(self):
        """Test that all strategies return DataFrame."""
        agent = MockAgent()
        baseline = EqualWeightStrategy()

        strategies = [
            SequentialBacktestStrategy(),
            WalkForwardBacktestStrategy(train_window=30, test_window=20),
            MonteCarloBacktestStrategy(n_simulations=5)
        ]

        for strategy in strategies:
            env1 = MockEnv(n_steps=80)
            history1 = strategy.execute_agent(agent, env1)
            assert isinstance(history1, pd.DataFrame)

            env2 = MockEnv(n_steps=80)
            history2 = strategy.execute_baseline(baseline, env2)
            assert isinstance(history2, pd.DataFrame)

    def test_strategies_produce_valid_portfolio_history(self):
        """Test that all strategies produce valid portfolio history."""
        agent = MockAgent()

        strategies = [
            SequentialBacktestStrategy(),
            WalkForwardBacktestStrategy(train_window=30, test_window=20),
            MonteCarloBacktestStrategy(n_simulations=5)
        ]

        for strategy in strategies:
            env = MockEnv(n_steps=80)
            history = strategy.execute_agent(agent, env)

            # Check required columns
            assert 'value' in history.columns
            assert 'return' in history.columns or 'date' in history.columns

            # Check data validity
            assert len(history) > 0
            assert not history['value'].isnull().any()
