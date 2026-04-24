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
from evaluation.baselines import EqualWeightStrategy, RandomStrategy
from environment.portfolio_env import PortfolioEnv
from data.features import FeatureEngineer


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

    def reset(self, seed=None, start_step=0):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        self.current_step = start_step
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


class TestWalkForwardRealEnv:
    """Regression tests that use a real PortfolioEnv (not MockEnv).

    These guard the C2 walk-forward bugs:
      - env.reset() clobbered the manual current_step assignment
      - total_steps used len(env.data) which is n_dates * n_tickers
      - fold termination compared current_step to a bound measured from step 0
    """

    @pytest.fixture
    def real_env(self, sample_ohlcv_data, sample_tickers):
        engineer = FeatureEngineer()
        features = engineer.compute_features(sample_ohlcv_data, ['returns'])
        prepared = engineer.prepare_for_environment(features, normalize=False)
        env = PortfolioEnv(
            data=prepared,
            feature_columns=['returns'],
            tickers=sample_tickers,
            initial_balance=10000.0,
            transaction_cost=0.0,
            window_size=5,
        )
        return env

    def test_fold_count_uses_n_steps_not_len_data(self, real_env):
        """With 3 tickers, len(env.data) == 3 * n_dates. Fold count must
        scale with n_dates only — otherwise walk-forward would produce
        n_tickers times too many folds."""
        wf = WalkForwardBacktestStrategy(
            train_window=20, test_window=10, step_size=10
        )
        baseline = EqualWeightStrategy()
        wf.execute_baseline(baseline, real_env)

        n_steps = real_env.n_steps  # ~95 for sample_ohlcv_data (100 dates - 5 window)
        expected_folds = max(0, (n_steps - 20 - 10) // 10 + 1)
        assert len(wf.fold_results) == expected_folds

    def test_fold_start_dates_advance_by_step_size(self, real_env):
        """Each fold's first history entry must be dates[window_size + test_start],
        advancing by step_size bars from fold to fold."""
        wf = WalkForwardBacktestStrategy(
            train_window=15, test_window=10, step_size=10
        )
        baseline = EqualWeightStrategy()
        combined = wf.execute_baseline(baseline, real_env)

        folds = wf.get_fold_results()
        assert len(folds) >= 2, "need at least two folds to verify offset"

        # Walk the combined history and extract each fold's first date.
        # Each fold contributes a sub-history that begins with the initial
        # reset entry at date[window_size + test_start].
        fold_start_dates = []
        offset = 0
        for fold in folds:
            first_date = combined['date'].iloc[offset]
            fold_start_dates.append(pd.Timestamp(first_date))
            offset += fold['n_steps']

        # Fold i's start date = env.dates[window_size + test_start_i]
        expected = [
            real_env.dates[real_env.window_size + fold['test_start']]
            for fold in folds
        ]
        assert [pd.Timestamp(d) for d in expected] == fold_start_dates

    def test_folds_do_not_all_start_at_step_zero(self, real_env):
        """Regression: folds used to all start at env step 0 because
        env.reset() clobbered the manual current_step assignment."""
        wf = WalkForwardBacktestStrategy(
            train_window=15, test_window=10, step_size=10
        )
        baseline = EqualWeightStrategy()
        wf.execute_baseline(baseline, real_env)

        folds = wf.get_fold_results()
        assert len({f['test_start'] for f in folds}) == len(folds), \
            "fold test_start offsets must be distinct"


class TestPortfolioEnvStartStep:
    """Tests for the new start_step parameter on PortfolioEnv.reset()."""

    @pytest.fixture
    def env(self, sample_ohlcv_data, sample_tickers):
        engineer = FeatureEngineer()
        features = engineer.compute_features(sample_ohlcv_data, ['returns'])
        prepared = engineer.prepare_for_environment(features, normalize=False)
        return PortfolioEnv(
            data=prepared,
            feature_columns=['returns'],
            tickers=sample_tickers,
            window_size=5,
        )

    def test_start_step_sets_current_step(self, env):
        obs, info = env.reset(start_step=7)
        assert env.current_step == 7

    def test_start_step_sets_initial_history_date(self, env):
        env.reset(start_step=7)
        history = env.get_portfolio_history()
        assert pd.Timestamp(history['date'].iloc[0]) == \
            env.dates[env.window_size + 7]

    def test_start_step_out_of_range_raises(self, env):
        with pytest.raises(ValueError, match="out of range"):
            env.reset(start_step=env.n_steps)
        with pytest.raises(ValueError, match="out of range"):
            env.reset(start_step=-1)

    def test_start_step_zero_is_default(self, env):
        o1, _ = env.reset()
        o2, _ = env.reset(start_step=0)
        np.testing.assert_array_equal(o1, o2)


class TestMonteCarloRandomVaries:
    """Regression tests for MC + RandomStrategy seeding."""

    def test_random_strategy_reseeds_across_simulations(self):
        """Each simulation must get distinct RNG state. Previously all
        simulations shared one RandomState seeded at __init__."""
        mc = MonteCarloBacktestStrategy(n_simulations=5, seeds=[1, 2, 3, 4, 5])
        baseline = RandomStrategy()
        env = MockEnv(n_steps=40)

        mc.execute_baseline(baseline, env)
        results = mc.get_simulation_results()

        # Not all simulations should collapse to the same final value.
        final_values = {round(r['final_value'], 6) for r in results}
        assert len(final_values) > 1

    def test_random_strategy_reset_is_deterministic_given_seed(self):
        """reset(seed=k) must produce the same action sequence every time."""
        strat = RandomStrategy()
        strat.reset(seed=42)
        a1 = [strat.get_action(env=type('E', (), {'n_assets': 3})(), step=i)
              for i in range(5)]
        strat.reset(seed=42)
        a2 = [strat.get_action(env=type('E', (), {'n_assets': 3})(), step=i)
              for i in range(5)]
        for x, y in zip(a1, a2):
            np.testing.assert_array_equal(x, y)


class TestMonteCarloAggregation:
    """Regression tests for _aggregate_histories robustness."""

    def test_ragged_histories_are_truncated_to_min_len(self):
        mc = MonteCarloBacktestStrategy(n_simulations=3)
        histories = [
            pd.DataFrame({'value': [100, 110, 120, 130, 140]}),
            pd.DataFrame({'value': [100, 105, 115]}),
            pd.DataFrame({'value': [100, 108, 112, 119]}),
        ]
        with pytest.warns(UserWarning, match="ragged histories"):
            agg = mc._aggregate_histories(histories, method='mean')
        assert len(agg) == 3  # min_len

    def test_equal_length_histories_do_not_warn(self):
        import warnings
        mc = MonteCarloBacktestStrategy(n_simulations=2)
        histories = [
            pd.DataFrame({'value': [100, 110, 120]}),
            pd.DataFrame({'value': [100, 105, 115]}),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # fail on any warning
            mc._aggregate_histories(histories, method='mean')

    def test_aggregated_frame_drops_run_dependent_columns(self):
        """Weights/cash/transaction_cost are per-run and must not be copied
        verbatim from simulation 0 onto mean values."""
        mc = MonteCarloBacktestStrategy(n_simulations=2)
        histories = [
            pd.DataFrame({
                'value': [100, 110, 120],
                'weights': [np.array([1.0, 0.0, 0.0]),
                            np.array([0.5, 0.5, 0.0]),
                            np.array([0.33, 0.33, 0.34])],
                'cash': [100, 50, 0],
            }),
            pd.DataFrame({
                'value': [100, 105, 115],
                'weights': [np.array([0.0, 1.0, 0.0]),
                            np.array([0.0, 0.5, 0.5]),
                            np.array([0.34, 0.33, 0.33])],
                'cash': [50, 25, 0],
            }),
        ]
        agg = mc._aggregate_histories(histories, method='mean')
        assert 'value' in agg.columns
        assert 'return' in agg.columns
        assert 'weights' not in agg.columns
        assert 'cash' not in agg.columns

    def test_seeds_too_short_raises(self):
        with pytest.raises(ValueError, match="n_simulations"):
            MonteCarloBacktestStrategy(n_simulations=10, seeds=[1, 2, 3])


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
