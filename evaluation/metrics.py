"""
Performance metrics for portfolio evaluation.

Refactored using simple composition pattern with focused metric calculators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class BaseMetrics:
    """Base class with common utility methods."""

    @staticmethod
    def _get_returns(portfolio_history: pd.DataFrame) -> np.ndarray:
        """Extract or compute returns from portfolio history."""
        if 'return' in portfolio_history.columns:
            return portfolio_history['return'].values
        else:
            values = portfolio_history['value'].values
            return np.diff(values) / values[:-1]


class ReturnMetrics(BaseMetrics):
    """Calculate return-based performance metrics."""

    def calculate(self, values: np.ndarray, n_periods: int, periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate return metrics.

        Args:
            values: Portfolio values over time
            n_periods: Number of periods
            periods_per_year: Trading periods per year (default: 252)

        Returns:
            Dictionary of return metrics
        """
        return {
            'total_return': self.total_return(values),
            'annualized_return': self.annualized_return(values, n_periods, periods_per_year)
        }

    @staticmethod
    def total_return(values: np.ndarray) -> float:
        """Calculate total return."""
        return (values[-1] / values[0]) - 1.0

    @staticmethod
    def annualized_return(values: np.ndarray, n_periods: int, periods_per_year: int = 252) -> float:
        """Calculate annualized return."""
        total_return = (values[-1] / values[0]) - 1.0
        n_years = n_periods / periods_per_year
        if n_years > 0:
            return (1 + total_return) ** (1 / n_years) - 1
        return 0.0


class RiskMetrics(BaseMetrics):
    """Calculate risk-adjusted performance metrics."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 252

    def calculate(self, returns: np.ndarray, periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate risk metrics.

        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year (default: 252)

        Returns:
            Dictionary of risk metrics
        """
        return {
            'sharpe_ratio': self.sharpe_ratio(returns, periods_per_year),
            'sortino_ratio': self.sortino_ratio(returns, periods_per_year),
            'volatility': self.volatility(returns, periods_per_year),
            'downside_deviation': self.downside_deviation(returns, periods_per_year)
        }

    def sharpe_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - self.daily_rf_rate
        if len(excess_returns) > 1 and np.std(excess_returns) > 0:
            return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
        return 0.0

    def sortino_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - self.daily_rf_rate
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std
        return 0.0

    @staticmethod
    def volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility."""
        if len(returns) > 1:
            return np.std(returns) * np.sqrt(periods_per_year)
        return 0.0

    def downside_deviation(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate annualized downside deviation."""
        excess_returns = returns - self.daily_rf_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            return np.std(downside_returns) * np.sqrt(periods_per_year)
        return 0.0


class DrawdownMetrics(BaseMetrics):
    """Calculate drawdown-based metrics."""

    def calculate(self, values: np.ndarray, n_periods: int, periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate drawdown metrics.

        Args:
            values: Portfolio values over time
            n_periods: Number of periods
            periods_per_year: Trading periods per year (default: 252)

        Returns:
            Dictionary of drawdown metrics
        """
        return {
            'max_drawdown': self.max_drawdown(values),
            'calmar_ratio': self.calmar_ratio(values, n_periods, periods_per_year)
        }

    @staticmethod
    def max_drawdown(values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return np.min(drawdown)

    def calmar_ratio(self, values: np.ndarray, n_periods: int, periods_per_year: int = 252) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        ann_return = ReturnMetrics.annualized_return(values, n_periods, periods_per_year)
        max_dd = abs(self.max_drawdown(values))
        if max_dd > 0:
            return ann_return / max_dd
        return 0.0


class WinLossMetrics(BaseMetrics):
    """Calculate win/loss statistics."""

    def calculate(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate win/loss metrics.

        Args:
            returns: Array of returns

        Returns:
            Dictionary of win/loss metrics
        """
        return {
            'win_rate': self.win_rate(returns),
            'avg_win': self.avg_win(returns),
            'avg_loss': self.avg_loss(returns),
            'profit_factor': self.profit_factor(returns)
        }

    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """Calculate win rate (proportion of positive returns)."""
        if len(returns) > 0:
            return np.sum(returns > 0) / len(returns)
        return 0.0

    @staticmethod
    def avg_win(returns: np.ndarray) -> float:
        """Calculate average winning return."""
        wins = returns[returns > 0]
        if len(wins) > 0:
            return np.mean(wins)
        return 0.0

    @staticmethod
    def avg_loss(returns: np.ndarray) -> float:
        """Calculate average losing return (as positive number)."""
        losses = returns[returns < 0]
        if len(losses) > 0:
            return abs(np.mean(losses))
        return 0.0

    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (sum of wins / sum of losses)."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        total_wins = np.sum(wins) if len(wins) > 0 else 0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0

        if total_losses > 0:
            return total_wins / total_losses
        return 0.0


class TransactionCostMetrics(BaseMetrics):
    """Calculate transaction cost metrics."""

    def calculate(self, portfolio_history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate transaction cost metrics.

        Args:
            portfolio_history: Portfolio history DataFrame

        Returns:
            Dictionary of transaction cost metrics (empty if no costs tracked)
        """
        if 'transaction_cost' not in portfolio_history.columns:
            return {}

        return {
            'total_transaction_costs': portfolio_history['transaction_cost'].sum(),
            'avg_transaction_cost': portfolio_history['transaction_cost'].mean()
        }


class BaselineComparisonMetrics(BaseMetrics):
    """Calculate metrics comparing portfolio to baseline."""

    def calculate(self, returns: np.ndarray, baseline_returns: np.ndarray,
                  periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate baseline comparison metrics.

        Args:
            returns: Portfolio returns
            baseline_returns: Baseline returns
            periods_per_year: Trading periods per year (default: 252)

        Returns:
            Dictionary of comparison metrics
        """
        return {
            'alpha': self.alpha(returns, baseline_returns, periods_per_year),
            'beta': self.beta(returns, baseline_returns),
            'information_ratio': self.information_ratio(returns, baseline_returns, periods_per_year)
        }

    @staticmethod
    def alpha(returns: np.ndarray, baseline_returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate alpha vs baseline."""
        if len(returns) != len(baseline_returns):
            raise ValueError("Returns and baseline must have same length")

        excess_returns = returns - baseline_returns
        return np.mean(excess_returns) * periods_per_year

    @staticmethod
    def beta(returns: np.ndarray, baseline_returns: np.ndarray) -> float:
        """Calculate beta vs baseline."""
        if len(returns) != len(baseline_returns):
            raise ValueError("Returns and baseline must have same length")

        if len(returns) > 1 and np.var(baseline_returns) > 0:
            covariance = np.cov(returns, baseline_returns)[0, 1]
            variance = np.var(baseline_returns)
            return covariance / variance
        return 0.0

    @staticmethod
    def information_ratio(returns: np.ndarray, baseline_returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate information ratio vs baseline."""
        if len(returns) != len(baseline_returns):
            raise ValueError("Returns and baseline must have same length")

        excess_returns = returns - baseline_returns
        if len(excess_returns) > 1 and np.std(excess_returns) > 0:
            return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
        return 0.0


class PortfolioMetrics:
    """
    Facade for calculating all portfolio performance metrics.

    Uses composition to delegate to specialized metric calculators.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
        """
        self.risk_free_rate = risk_free_rate

        # Compose specialized calculators
        self.return_metrics = ReturnMetrics()
        self.risk_metrics = RiskMetrics(risk_free_rate)
        self.drawdown_metrics = DrawdownMetrics()
        self.win_loss_metrics = WinLossMetrics()
        self.cost_metrics = TransactionCostMetrics()
        self.baseline_metrics = BaselineComparisonMetrics()

    def calculate_all_metrics(
        self,
        portfolio_history: pd.DataFrame,
        baseline_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        Args:
            portfolio_history: DataFrame with portfolio history (from env.get_portfolio_history())
            baseline_returns: Optional baseline returns for comparison

        Returns:
            Dictionary of metrics
        """
        values = portfolio_history['value'].values
        returns = BaseMetrics._get_returns(portfolio_history)
        n_periods = len(values)

        # Aggregate metrics from all calculators
        metrics = {}
        metrics.update(self.return_metrics.calculate(values, n_periods))
        metrics.update(self.risk_metrics.calculate(returns))
        metrics.update(self.drawdown_metrics.calculate(values, n_periods))
        metrics.update(self.win_loss_metrics.calculate(returns))
        metrics.update(self.cost_metrics.calculate(portfolio_history))

        # Add baseline comparison if provided
        if baseline_returns is not None:
            baseline_array = baseline_returns.values if isinstance(baseline_returns, pd.Series) else baseline_returns
            metrics.update(self.baseline_metrics.calculate(returns, baseline_array))

        return metrics

    def print_metrics(self, metrics: Dict[str, float]):
        """Pretty print metrics."""
        print("\n" + "="*60)
        print("PORTFOLIO PERFORMANCE METRICS")
        print("="*60)

        print("\nReturns:")
        print(f"  Total Return:        {metrics['total_return']:>10.2%}")
        print(f"  Annualized Return:   {metrics['annualized_return']:>10.2%}")

        print("\nRisk-Adjusted:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.4f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.4f}")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>10.4f}")

        print("\nRisk:")
        print(f"  Volatility:          {metrics['volatility']:>10.2%}")
        print(f"  Downside Deviation:  {metrics['downside_deviation']:>10.2%}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:>10.2%}")

        print("\nWin/Loss:")
        print(f"  Win Rate:            {metrics['win_rate']:>10.2%}")
        print(f"  Avg Win:             {metrics['avg_win']:>10.4f}")
        print(f"  Avg Loss:            {metrics['avg_loss']:>10.4f}")
        print(f"  Profit Factor:       {metrics['profit_factor']:>10.4f}")

        if 'total_transaction_costs' in metrics:
            print("\nTransaction Costs:")
            print(f"  Total Costs:         ${metrics['total_transaction_costs']:>10.2f}")
            print(f"  Avg Cost per Trade:  ${metrics['avg_transaction_cost']:>10.4f}")

        if 'alpha' in metrics:
            print("\nVs Baseline:")
            print(f"  Alpha:               {metrics['alpha']:>10.2%}")
            print(f"  Beta:                {metrics['beta']:>10.4f}")
            print(f"  Information Ratio:   {metrics['information_ratio']:>10.4f}")

        print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    # Create sample portfolio history
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 252)
    values = 10000 * np.cumprod(1 + returns)

    history = pd.DataFrame({
        'date': dates,
        'value': values,
        'return': returns
    })

    # Calculate metrics
    calculator = PortfolioMetrics(risk_free_rate=0.02)
    metrics = calculator.calculate_all_metrics(history)

    # Print results
    calculator.print_metrics(metrics)
