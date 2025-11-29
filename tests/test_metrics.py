"""
Unit tests for evaluation/metrics.py

Tests the composition pattern with individual metric calculators.
"""

import pytest
import numpy as np
import pandas as pd
from evaluation.metrics import (
    PortfolioMetrics,
    ReturnMetrics,
    RiskMetrics,
    DrawdownMetrics,
    WinLossMetrics,
    TransactionCostMetrics,
    BaselineComparisonMetrics,
    BaseMetrics
)


class TestReturnMetrics:
    """Tests for ReturnMetrics calculator."""

    def test_total_return(self):
        """Test total return calculation."""
        values = np.array([10000, 10500, 11000, 10800, 11500])
        calc = ReturnMetrics()
        metrics = calc.calculate(values, n_periods=len(values))

        expected = (11500 / 10000) - 1.0
        assert np.isclose(metrics['total_return'], expected, atol=1e-6)

    def test_total_return_with_loss(self):
        """Test total return with loss."""
        values = np.array([10000, 9500, 9000])
        calc = ReturnMetrics()
        metrics = calc.calculate(values, n_periods=len(values))

        assert metrics['total_return'] < 0
        assert np.isclose(metrics['total_return'], -0.1, atol=1e-6)

    def test_annualized_return(self):
        """Test annualized return calculation."""
        values = np.array([10000] + [10000 * (1.1 ** (i / 252)) for i in range(1, 253)])
        calc = ReturnMetrics()
        metrics = calc.calculate(values, n_periods=252)

        assert metrics['annualized_return'] > 0
        assert np.isclose(metrics['annualized_return'], 0.1, atol=0.01)

    def test_annualized_return_zero_periods(self):
        """Test annualized return with zero periods."""
        values = np.array([10000, 10500])
        calc = ReturnMetrics()
        metrics = calc.calculate(values, n_periods=0)

        assert metrics['annualized_return'] == 0.0


class TestRiskMetrics:
    """Tests for RiskMetrics calculator."""

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        calc = RiskMetrics(risk_free_rate=0.02)
        metrics = calc.calculate(returns)

        assert isinstance(metrics['sharpe_ratio'], float)
        assert 'volatility' in metrics
        assert 'sortino_ratio' in metrics

    def test_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation."""
        returns = np.ones(100) * 0.001
        calc = RiskMetrics(risk_free_rate=0.02)
        metrics = calc.calculate(returns)

        assert isinstance(metrics['sharpe_ratio'], float)

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        calc = RiskMetrics(risk_free_rate=0.02)
        metrics = calc.calculate(returns)

        assert isinstance(metrics['sortino_ratio'], float)

    def test_sortino_ratio_no_downside(self):
        """Test Sortino ratio with all positive returns."""
        returns = np.abs(np.random.normal(0.001, 0.01, 252))
        calc = RiskMetrics(risk_free_rate=0.02)
        metrics = calc.calculate(returns)

        assert metrics['sortino_ratio'] == 0.0

    def test_volatility(self):
        """Test volatility calculation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        calc = RiskMetrics(risk_free_rate=0.02)
        metrics = calc.calculate(returns)

        assert metrics['volatility'] > 0
        assert isinstance(metrics['volatility'], float)

    def test_volatility_single_return(self):
        """Test volatility with single return."""
        returns = np.array([0.01])
        calc = RiskMetrics(risk_free_rate=0.02)
        metrics = calc.calculate(returns)

        assert metrics['volatility'] == 0.0

    def test_downside_deviation(self):
        """Test downside deviation calculation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        calc = RiskMetrics(risk_free_rate=0.02)
        metrics = calc.calculate(returns)

        assert metrics['downside_deviation'] >= 0
        assert isinstance(metrics['downside_deviation'], float)


class TestDrawdownMetrics:
    """Tests for DrawdownMetrics calculator."""

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        values = np.array([10000, 11000, 10500, 10000, 9500, 10500, 11500])
        calc = DrawdownMetrics()
        metrics = calc.calculate(values, n_periods=len(values))

        assert metrics['max_drawdown'] < 0
        expected = (9500 - 11000) / 11000
        assert np.isclose(metrics['max_drawdown'], expected, atol=1e-6)

    def test_max_drawdown_always_increasing(self):
        """Test max drawdown with always increasing portfolio."""
        values = np.array([10000, 11000, 12000, 13000])
        calc = DrawdownMetrics()
        metrics = calc.calculate(values, n_periods=len(values))

        assert metrics['max_drawdown'] == 0.0

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        values = np.array([10000] + [10000 * (1.1 ** (i / 252)) for i in range(1, 253)])
        calc = DrawdownMetrics()
        metrics = calc.calculate(values, n_periods=252)

        assert isinstance(metrics['calmar_ratio'], float)
        assert 'max_drawdown' in metrics

    def test_calmar_ratio_zero_drawdown(self):
        """Test Calmar ratio with zero drawdown."""
        values = np.array([10000, 11000, 12000])
        calc = DrawdownMetrics()
        metrics = calc.calculate(values, n_periods=len(values))

        assert metrics['calmar_ratio'] == 0.0


class TestWinLossMetrics:
    """Tests for WinLossMetrics calculator."""

    def test_win_rate(self):
        """Test win rate calculation."""
        returns = np.array([0.01, -0.01, 0.02, -0.005, 0.015])
        calc = WinLossMetrics()
        metrics = calc.calculate(returns)

        expected = 3 / 5
        assert np.isclose(metrics['win_rate'], expected, atol=1e-6)

    def test_win_rate_all_wins(self):
        """Test win rate with all positive returns."""
        returns = np.array([0.01, 0.02, 0.015])
        calc = WinLossMetrics()
        metrics = calc.calculate(returns)

        assert metrics['win_rate'] == 1.0

    def test_win_rate_all_losses(self):
        """Test win rate with all negative returns."""
        returns = np.array([-0.01, -0.02, -0.015])
        calc = WinLossMetrics()
        metrics = calc.calculate(returns)

        assert metrics['win_rate'] == 0.0

    def test_avg_win(self):
        """Test average win calculation."""
        returns = np.array([0.01, -0.01, 0.02, -0.005, 0.015])
        calc = WinLossMetrics()
        metrics = calc.calculate(returns)

        expected = (0.01 + 0.02 + 0.015) / 3
        assert np.isclose(metrics['avg_win'], expected, atol=1e-6)

    def test_avg_win_no_wins(self):
        """Test average win with no winning returns."""
        returns = np.array([-0.01, -0.02])
        calc = WinLossMetrics()
        metrics = calc.calculate(returns)

        assert metrics['avg_win'] == 0.0

    def test_avg_loss(self):
        """Test average loss calculation."""
        returns = np.array([0.01, -0.01, 0.02, -0.005, 0.015])
        calc = WinLossMetrics()
        metrics = calc.calculate(returns)

        expected = (0.01 + 0.005) / 2
        assert np.isclose(metrics['avg_loss'], expected, atol=1e-6)

    def test_avg_loss_no_losses(self):
        """Test average loss with no losing returns."""
        returns = np.array([0.01, 0.02])
        calc = WinLossMetrics()
        metrics = calc.calculate(returns)

        assert metrics['avg_loss'] == 0.0

    def test_profit_factor(self):
        """Test profit factor calculation."""
        returns = np.array([0.02, -0.01, 0.03, -0.01])
        calc = WinLossMetrics()
        metrics = calc.calculate(returns)

        total_wins = 0.02 + 0.03
        total_losses = 0.01 + 0.01
        expected = total_wins / total_losses

        assert np.isclose(metrics['profit_factor'], expected, atol=1e-6)

    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        returns = np.array([0.01, 0.02, 0.03])
        calc = WinLossMetrics()
        metrics = calc.calculate(returns)

        assert metrics['profit_factor'] == 0.0


class TestTransactionCostMetrics:
    """Tests for TransactionCostMetrics calculator."""

    def test_transaction_costs_present(self):
        """Test transaction cost calculation when costs are tracked."""
        history = pd.DataFrame({
            'transaction_cost': [1.0, 2.0, 1.5, 3.0, 2.5]
        })
        calc = TransactionCostMetrics()
        metrics = calc.calculate(history)

        assert metrics['total_transaction_costs'] == 10.0
        assert metrics['avg_transaction_cost'] == 2.0

    def test_transaction_costs_absent(self):
        """Test transaction cost calculation when costs are not tracked."""
        history = pd.DataFrame({
            'value': [10000, 10500, 11000]
        })
        calc = TransactionCostMetrics()
        metrics = calc.calculate(history)

        assert metrics == {}


class TestBaselineComparisonMetrics:
    """Tests for BaselineComparisonMetrics calculator."""

    def test_alpha(self):
        """Test alpha calculation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)
        baseline_returns = np.random.normal(0.0005, 0.01, 252)

        calc = BaselineComparisonMetrics()
        metrics = calc.calculate(returns, baseline_returns)

        assert isinstance(metrics['alpha'], float)
        assert 'beta' in metrics
        assert 'information_ratio' in metrics

    def test_alpha_mismatched_length_raises_error(self):
        """Test that mismatched lengths raise error."""
        returns = np.array([0.01, 0.02])
        baseline_returns = np.array([0.01])

        calc = BaselineComparisonMetrics()

        with pytest.raises(ValueError, match="Returns and baseline must have same length"):
            calc.calculate(returns, baseline_returns)

    def test_beta(self):
        """Test beta calculation."""
        np.random.seed(42)
        baseline_returns = np.random.normal(0.001, 0.01, 252)
        returns = baseline_returns * 1.5 + np.random.normal(0, 0.005, 252)

        calc = BaselineComparisonMetrics()
        metrics = calc.calculate(returns, baseline_returns)

        assert isinstance(metrics['beta'], float)
        assert metrics['beta'] > 0

    def test_beta_zero_variance(self):
        """Test beta with zero variance baseline."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 100)
        baseline_returns = np.ones(100) * 0.001

        calc = BaselineComparisonMetrics()
        metrics = calc.calculate(returns, baseline_returns)

        assert isinstance(metrics['beta'], float)

    def test_information_ratio(self):
        """Test information ratio calculation."""
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.01, 252)
        baseline_returns = np.random.normal(0.001, 0.01, 252)

        calc = BaselineComparisonMetrics()
        metrics = calc.calculate(returns, baseline_returns)

        assert isinstance(metrics['information_ratio'], float)


class TestPortfolioMetrics:
    """Tests for PortfolioMetrics facade (composition)."""

    @pytest.fixture
    def metrics_calculator(self):
        """Create a metrics calculator."""
        return PortfolioMetrics(risk_free_rate=0.02)

    def test_composition_returns_all_calculator_instances(self, metrics_calculator):
        """Test that PortfolioMetrics composes all calculators."""
        assert isinstance(metrics_calculator.return_metrics, ReturnMetrics)
        assert isinstance(metrics_calculator.risk_metrics, RiskMetrics)
        assert isinstance(metrics_calculator.drawdown_metrics, DrawdownMetrics)
        assert isinstance(metrics_calculator.win_loss_metrics, WinLossMetrics)
        assert isinstance(metrics_calculator.cost_metrics, TransactionCostMetrics)
        assert isinstance(metrics_calculator.baseline_metrics, BaselineComparisonMetrics)

    def test_calculate_all_metrics(self, sample_portfolio_history, metrics_calculator):
        """Test calculating all metrics at once via facade."""
        metrics = metrics_calculator.calculate_all_metrics(sample_portfolio_history)

        assert isinstance(metrics, dict)

        # Check return metrics
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics

        # Check risk metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'volatility' in metrics
        assert 'downside_deviation' in metrics

        # Check drawdown metrics
        assert 'max_drawdown' in metrics
        assert 'calmar_ratio' in metrics

        # Check win/loss metrics
        assert 'win_rate' in metrics
        assert 'avg_win' in metrics
        assert 'avg_loss' in metrics
        assert 'profit_factor' in metrics

        # Check transaction cost metrics
        assert 'total_transaction_costs' in metrics
        assert 'avg_transaction_cost' in metrics

    def test_calculate_all_metrics_with_baseline(self, sample_portfolio_history, metrics_calculator):
        """Test calculating metrics with baseline comparison."""
        baseline_returns = np.random.normal(0.0005, 0.01, len(sample_portfolio_history))

        metrics = metrics_calculator.calculate_all_metrics(
            sample_portfolio_history,
            baseline_returns=baseline_returns
        )

        # Check baseline comparison metrics
        assert 'alpha' in metrics
        assert 'beta' in metrics
        assert 'information_ratio' in metrics

    def test_calculate_all_metrics_without_transaction_costs(self, metrics_calculator):
        """Test metrics calculation when transaction costs are absent."""
        history = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'value': 10000 * np.cumprod(1 + np.random.normal(0.001, 0.01, 100)),
            'return': np.random.normal(0.001, 0.01, 100)
        })

        metrics = metrics_calculator.calculate_all_metrics(history)

        # Transaction cost metrics should not be present
        assert 'total_transaction_costs' not in metrics
        assert 'avg_transaction_cost' not in metrics

        # But other metrics should be there
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics

    def test_print_metrics_does_not_raise(self, sample_portfolio_history, metrics_calculator):
        """Test that print_metrics runs without error."""
        metrics = metrics_calculator.calculate_all_metrics(sample_portfolio_history)
        metrics_calculator.print_metrics(metrics)


class TestBaseMetrics:
    """Tests for BaseMetrics utility methods."""

    def test_get_returns_from_return_column(self):
        """Test getting returns from return column."""
        history = pd.DataFrame({
            'value': [10000, 10500, 11000],
            'return': [0.0, 0.05, 0.0476]
        })

        returns = BaseMetrics._get_returns(history)

        assert len(returns) == 3
        assert np.isclose(returns[0], 0.0)
        assert np.isclose(returns[1], 0.05, atol=1e-4)

    def test_get_returns_computed_from_values(self):
        """Test computing returns from values when return column absent."""
        history = pd.DataFrame({
            'value': [10000, 10500, 11000]
        })

        returns = BaseMetrics._get_returns(history)

        assert len(returns) == 2  # n-1 returns from diff
        expected = np.array([0.05, (11000 - 10500) / 10500])
        assert np.allclose(returns, expected, atol=1e-6)
