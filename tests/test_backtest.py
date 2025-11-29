"""
Unit tests for evaluation/backtest.py
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from unittest.mock import MagicMock
from evaluation.backtest import BacktestResult, Backtester, run_agent_backtest, run_baseline_backtest


class TestBacktestResult:
    """Tests for BacktestResult class."""

    def test_init(self, sample_portfolio_history):
        """Test BacktestResult initialization."""
        result = BacktestResult(
            name='test_strategy',
            history=sample_portfolio_history,
            metadata={'type': 'baseline'}
        )

        assert result.name == 'test_strategy'
        assert isinstance(result.history, pd.DataFrame)
        assert result.metadata['type'] == 'baseline'

    def test_init_without_metadata(self, sample_portfolio_history):
        """Test initialization without metadata."""
        result = BacktestResult(
            name='test_strategy',
            history=sample_portfolio_history
        )

        assert result.metadata == {}

    def test_repr(self, sample_portfolio_history):
        """Test string representation."""
        result = BacktestResult(
            name='test_strategy',
            history=sample_portfolio_history
        )

        repr_str = repr(result)
        assert 'test_strategy' in repr_str
        assert str(len(sample_portfolio_history)) in repr_str


class TestBacktester:
    """Tests for Backtester class."""

    @pytest.fixture
    def backtester(self):
        """Create a backtester instance."""
        return Backtester(risk_free_rate=0.02)

    @pytest.fixture
    def mock_env(self, sample_portfolio_history):
        """Create a mock environment."""
        env = MagicMock()
        env.reset.return_value = (np.zeros(10), {})
        env.step.return_value = (
            np.zeros(10),
            0.01,
            False,
            False,
            {'portfolio_value': 10100}
        )
        env.get_portfolio_history.return_value = sample_portfolio_history
        return env

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock()
        agent.predict.return_value = (np.array([1.0, 1.0, 1.0]), None)
        return agent

    def test_init(self, backtester):
        """Test backtester initialization."""
        assert backtester.risk_free_rate == 0.02
        assert len(backtester.results) == 0
        assert backtester.baseline_registry is not None
        assert backtester.metrics_calculator is not None

    def test_run_agent_stores_result(self, backtester, mock_agent, mock_env):
        """Test that run_agent stores result."""
        mock_env.step.return_value = (
            np.zeros(10),
            0.01,
            True,
            False,
            {'portfolio_value': 10100}
        )

        result = backtester.run_agent(mock_agent, mock_env, name='test_agent')

        assert isinstance(result, BacktestResult)
        assert result.name == 'test_agent'
        assert 'test_agent' in backtester.results

    def test_run_baseline_stores_result(self, backtester, mock_env):
        """Test that run_baseline stores result."""
        mock_env.step.return_value = (
            np.zeros(10),
            0.01,
            True,
            False,
            {'portfolio_value': 10100}
        )
        mock_env.n_assets = 3

        result = backtester.run_baseline(mock_env, 'equal_weight')

        assert isinstance(result, BacktestResult)
        assert result.name == 'equal_weight'
        assert 'equal_weight' in backtester.results

    def test_run_baseline_with_custom_name(self, backtester, mock_env):
        """Test run_baseline with custom result name."""
        mock_env.step.return_value = (
            np.zeros(10),
            0.01,
            True,
            False,
            {'portfolio_value': 10100}
        )
        mock_env.n_assets = 3

        result = backtester.run_baseline(
            mock_env,
            'equal_weight',
            result_name='custom_name'
        )

        assert result.name == 'custom_name'
        assert 'custom_name' in backtester.results

    def test_add_result(self, backtester, sample_portfolio_history):
        """Test adding a pre-computed result."""
        result = BacktestResult('manual_result', sample_portfolio_history)
        backtester.add_result(result)

        assert 'manual_result' in backtester.results
        assert backtester.get_result('manual_result') == result

    def test_get_result(self, backtester, sample_portfolio_history):
        """Test getting a stored result."""
        result = BacktestResult('test', sample_portfolio_history)
        backtester.add_result(result)

        retrieved = backtester.get_result('test')
        assert retrieved == result

    def test_get_nonexistent_result_raises_error(self, backtester):
        """Test that getting nonexistent result raises error."""
        with pytest.raises(ValueError, match="Result .* not found"):
            backtester.get_result('nonexistent')

    def test_list_results(self, backtester, sample_portfolio_history):
        """Test listing all results."""
        result1 = BacktestResult('strategy1', sample_portfolio_history)
        result2 = BacktestResult('strategy2', sample_portfolio_history)

        backtester.add_result(result1)
        backtester.add_result(result2)

        results = backtester.list_results()
        assert 'strategy1' in results
        assert 'strategy2' in results
        assert len(results) == 2

    def test_clear_results(self, backtester, sample_portfolio_history):
        """Test clearing all results."""
        result = BacktestResult('test', sample_portfolio_history)
        backtester.add_result(result)

        assert len(backtester.results) == 1

        backtester.clear_results()
        assert len(backtester.results) == 0

    def test_compute_metrics(self, backtester, sample_portfolio_history):
        """Test computing metrics for a result."""
        result = BacktestResult('test', sample_portfolio_history)
        backtester.add_result(result)

        metrics = backtester.compute_metrics('test')

        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics

    def test_compute_all_metrics(self, backtester, sample_portfolio_history):
        """Test computing metrics for all results."""
        result1 = BacktestResult('strategy1', sample_portfolio_history)
        result2 = BacktestResult('strategy2', sample_portfolio_history)

        backtester.add_result(result1)
        backtester.add_result(result2)

        all_metrics = backtester.compute_all_metrics()

        assert isinstance(all_metrics, dict)
        assert 'strategy1' in all_metrics
        assert 'strategy2' in all_metrics
        assert 'total_return' in all_metrics['strategy1']

    def test_print_comparison(self, backtester, sample_portfolio_history):
        """Test that print_comparison runs without error."""
        result1 = BacktestResult('strategy1', sample_portfolio_history)
        result2 = BacktestResult('strategy2', sample_portfolio_history)

        backtester.add_result(result1)
        backtester.add_result(result2)

        backtester.print_comparison()

    def test_print_comparison_with_names(self, backtester, sample_portfolio_history):
        """Test print_comparison with specific names."""
        result1 = BacktestResult('strategy1', sample_portfolio_history)
        result2 = BacktestResult('strategy2', sample_portfolio_history)

        backtester.add_result(result1)
        backtester.add_result(result2)

        backtester.print_comparison(names=['strategy1'])

    def test_print_comparison_with_no_results(self, backtester):
        """Test print_comparison with no results."""
        backtester.print_comparison()

    def test_get_histories(self, backtester, sample_portfolio_history):
        """Test getting multiple histories."""
        result1 = BacktestResult('strategy1', sample_portfolio_history)
        result2 = BacktestResult('strategy2', sample_portfolio_history)

        backtester.add_result(result1)
        backtester.add_result(result2)

        histories = backtester.get_histories()

        assert isinstance(histories, dict)
        assert 'strategy1' in histories
        assert 'strategy2' in histories
        assert isinstance(histories['strategy1'], pd.DataFrame)

    def test_get_histories_with_names(self, backtester, sample_portfolio_history):
        """Test getting specific histories."""
        result1 = BacktestResult('strategy1', sample_portfolio_history)
        result2 = BacktestResult('strategy2', sample_portfolio_history)

        backtester.add_result(result1)
        backtester.add_result(result2)

        histories = backtester.get_histories(names=['strategy1'])

        assert len(histories) == 1
        assert 'strategy1' in histories

    def test_export_results(self, backtester, sample_portfolio_history):
        """Test exporting results to CSV."""
        temp_dir = tempfile.mkdtemp()

        try:
            result = BacktestResult('strategy1', sample_portfolio_history)
            backtester.add_result(result)

            backtester.export_results(temp_dir)

            import os
            assert os.path.exists(os.path.join(temp_dir, 'strategy1_history.csv'))
            assert os.path.exists(os.path.join(temp_dir, 'metrics_comparison.csv'))
        finally:
            shutil.rmtree(temp_dir)

    def test_export_specific_results(self, backtester, sample_portfolio_history):
        """Test exporting specific results."""
        temp_dir = tempfile.mkdtemp()

        try:
            result1 = BacktestResult('strategy1', sample_portfolio_history)
            result2 = BacktestResult('strategy2', sample_portfolio_history)

            backtester.add_result(result1)
            backtester.add_result(result2)

            backtester.export_results(temp_dir, names=['strategy1'])

            import os
            assert os.path.exists(os.path.join(temp_dir, 'strategy1_history.csv'))
            assert not os.path.exists(os.path.join(temp_dir, 'strategy2_history.csv'))
        finally:
            shutil.rmtree(temp_dir)


class TestStandaloneFunctions:
    """Tests for standalone backtest functions."""

    @pytest.fixture
    def mock_env_terminated(self, sample_portfolio_history):
        """Create a mock environment that terminates immediately."""
        env = MagicMock()
        env.reset.return_value = (np.zeros(10), {})
        env.step.return_value = (
            np.zeros(10),
            0.01,
            True,
            False,
            {'portfolio_value': 10100}
        )
        env.get_portfolio_history.return_value = sample_portfolio_history
        env.n_assets = 3
        return env

    def test_run_agent_backtest(self, mock_env_terminated):
        """Test standalone run_agent_backtest function."""
        agent = MagicMock()
        agent.predict.return_value = (np.array([1.0, 1.0, 1.0]), None)

        history = run_agent_backtest(agent, mock_env_terminated)

        assert isinstance(history, pd.DataFrame)
        assert 'value' in history.columns

    def test_run_agent_backtest_deterministic(self, mock_env_terminated):
        """Test run_agent_backtest with deterministic flag."""
        agent = MagicMock()
        agent.predict.return_value = (np.array([1.0, 1.0, 1.0]), None)

        history = run_agent_backtest(agent, mock_env_terminated, deterministic=True)

        assert isinstance(history, pd.DataFrame)
        # Check that predict was called with deterministic=True
        call_args = agent.predict.call_args
        assert call_args[1]['deterministic'] == True

    def test_run_baseline_backtest(self, mock_env_terminated):
        """Test standalone run_baseline_backtest function."""
        history = run_baseline_backtest(mock_env_terminated, 'equal_weight')

        assert isinstance(history, pd.DataFrame)
        assert 'value' in history.columns

    def test_run_baseline_backtest_unknown_strategy(self, mock_env_terminated):
        """Test run_baseline_backtest with unknown strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_baseline_backtest(mock_env_terminated, 'unknown_strategy')
