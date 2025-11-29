"""
Backtesting framework for portfolio strategies.
Stateful class that stores and analyzes backtest results.

Refactored to use Strategy pattern for different execution modes.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from .metrics import PortfolioMetrics
from .baselines import BaselineStrategyRegistry
from .backtest_strategies import (
    BacktestStrategy,
    SequentialBacktestStrategy,
    WalkForwardBacktestStrategy,
    MonteCarloBacktestStrategy
)


class BacktestResult:
    """Container for backtest results."""

    def __init__(self, name: str, history: pd.DataFrame, metadata: Optional[Dict] = None):
        """
        Initialize backtest result.

        Args:
            name: Strategy name
            history: Portfolio history DataFrame
            metadata: Optional metadata about the backtest
        """
        self.name = name
        self.history = history
        self.metadata = metadata or {}

    def __repr__(self):
        return f"BacktestResult(name='{self.name}', periods={len(self.history)})"


class Backtester:
    """
    Stateful backtester that stores results and provides analysis.

    Uses Strategy pattern for different execution modes (sequential, walk-forward, Monte Carlo).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        execution_strategy: Optional[BacktestStrategy] = None
    ):
        """
        Initialize backtester.

        Args:
            risk_free_rate: Annual risk-free rate
            execution_strategy: Backtest execution strategy (default: SequentialBacktestStrategy)
        """
        self.risk_free_rate = risk_free_rate
        self.metrics_calculator = PortfolioMetrics(risk_free_rate)
        self.baseline_registry = BaselineStrategyRegistry()

        # Execution strategy (Strategy pattern)
        self.execution_strategy = execution_strategy or SequentialBacktestStrategy()

        # Store results
        self.results: Dict[str, BacktestResult] = {}

    def set_execution_strategy(self, strategy: BacktestStrategy):
        """
        Change the execution strategy.

        Args:
            strategy: New backtest execution strategy
        """
        self.execution_strategy = strategy

    def run_agent(
        self,
        agent,
        env,
        name: str = 'agent',
        deterministic: bool = True
    ) -> BacktestResult:
        """
        Run backtest using a trained agent and store result.

        Uses the configured execution strategy.

        Args:
            agent: Trained SB3 agent
            env: Portfolio environment
            name: Name for this result
            deterministic: Whether to use deterministic actions

        Returns:
            BacktestResult object
        """
        # Use execution strategy
        history = self.execution_strategy.execute_agent(
            agent,
            env,
            deterministic=deterministic
        )

        result = BacktestResult(
            name=name,
            history=history,
            metadata={
                'type': 'agent',
                'deterministic': deterministic,
                'execution_strategy': self.execution_strategy.__class__.__name__
            }
        )

        self.results[name] = result
        return result

    def run_baseline(
        self,
        env,
        strategy_name: str,
        result_name: Optional[str] = None
    ) -> BacktestResult:
        """
        Run backtest using a baseline strategy and store result.

        Uses the configured execution strategy.

        Args:
            env: Portfolio environment
            strategy_name: Name of baseline strategy
            result_name: Optional custom name for result (defaults to strategy_name)

        Returns:
            BacktestResult object
        """
        strategy = self.baseline_registry.get(strategy_name)

        # Use execution strategy
        history = self.execution_strategy.execute_baseline(strategy, env)

        name = result_name or strategy_name
        result = BacktestResult(
            name=name,
            history=history,
            metadata={
                'type': 'baseline',
                'strategy': strategy_name,
                'execution_strategy': self.execution_strategy.__class__.__name__
            }
        )

        self.results[name] = result
        return result

    def add_result(self, result: BacktestResult):
        """
        Add a pre-computed result to the backtester.

        Args:
            result: BacktestResult to add
        """
        self.results[result.name] = result

    def get_result(self, name: str) -> BacktestResult:
        """Get a stored result by name."""
        if name not in self.results:
            raise ValueError(f"Result '{name}' not found. Available: {self.list_results()}")
        return self.results[name]

    def list_results(self) -> List[str]:
        """List all stored result names."""
        return list(self.results.keys())

    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()

    def compute_metrics(self, name: str) -> Dict[str, float]:
        """
        Compute metrics for a stored result.

        Args:
            name: Result name

        Returns:
            Dictionary of metrics
        """
        result = self.get_result(name)
        return self.metrics_calculator.calculate_all_metrics(result.history)

    def compute_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for all stored results.

        Returns:
            Dictionary mapping result names to their metrics
        """
        all_metrics = {}
        for name in self.results:
            all_metrics[name] = self.compute_metrics(name)
        return all_metrics

    def print_comparison(self, names: Optional[List[str]] = None):
        """
        Print comparison of strategies.

        Args:
            names: Optional list of result names to compare (defaults to all)
        """
        if names is None:
            names = self.list_results()

        if not names:
            print("No results to compare.")
            return

        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)

        all_metrics = {}
        for name in names:
            result = self.get_result(name)
            print(f"\n{name.upper()}:")
            metrics = self.metrics_calculator.calculate_all_metrics(result.history)
            all_metrics[name] = metrics
            self.metrics_calculator.print_metrics(metrics)

        # Print comparison table
        self._print_comparison_table(all_metrics)

    def _print_comparison_table(self, all_metrics: Dict[str, Dict[str, float]]):
        """Print comparison table of key metrics."""
        print("\n" + "="*80)
        print("KEY METRICS COMPARISON")
        print("="*80)

        # Select key metrics to display
        key_metrics = [
            'total_return', 'annualized_return', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'volatility'
        ]

        # Create comparison DataFrame
        comparison = pd.DataFrame(all_metrics).T[key_metrics]

        # Format for display
        format_dict = {
            'total_return': lambda x: f'{x:.2%}',
            'annualized_return': lambda x: f'{x:.2%}',
            'sharpe_ratio': lambda x: f'{x:.4f}',
            'sortino_ratio': lambda x: f'{x:.4f}',
            'max_drawdown': lambda x: f'{x:.2%}',
            'volatility': lambda x: f'{x:.2%}'
        }

        print("\n", comparison.to_string(
            formatters=format_dict,
            col_space=15
        ))
        print("\n" + "="*80)

    def get_histories(self, names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Get portfolio histories for multiple results.

        Args:
            names: Optional list of result names (defaults to all)

        Returns:
            Dictionary mapping names to history DataFrames
        """
        if names is None:
            names = self.list_results()

        return {name: self.get_result(name).history for name in names}

    def export_results(self, path: str, names: Optional[List[str]] = None):
        """
        Export results to CSV files.

        Args:
            path: Directory path to save results
            names: Optional list of result names to export (defaults to all)
        """
        import os
        os.makedirs(path, exist_ok=True)

        if names is None:
            names = self.list_results()

        for name in names:
            result = self.get_result(name)
            filepath = os.path.join(path, f"{name}_history.csv")
            result.history.to_csv(filepath, index=False)
            print(f"Exported {name} to {filepath}")

        # Export metrics comparison
        all_metrics = self.compute_all_metrics()
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_path = os.path.join(path, "metrics_comparison.csv")
        metrics_df.to_csv(metrics_path)
        print(f"Exported metrics to {metrics_path}")


# Standalone utility functions for one-off backtests
def run_agent_backtest(
    agent,
    env,
    deterministic: bool = True,
    execution_strategy: Optional[BacktestStrategy] = None
) -> pd.DataFrame:
    """
    Standalone function to run agent backtest without storing results.

    Args:
        agent: Trained SB3 agent
        env: Portfolio environment
        deterministic: Whether to use deterministic actions
        execution_strategy: Optional execution strategy (default: Sequential)

    Returns:
        Portfolio history DataFrame
    """
    strategy = execution_strategy or SequentialBacktestStrategy()
    return strategy.execute_agent(agent, env, deterministic=deterministic)


def run_baseline_backtest(
    env,
    strategy_name: str,
    execution_strategy: Optional[BacktestStrategy] = None
) -> pd.DataFrame:
    """
    Standalone function to run baseline backtest without storing results.

    Args:
        env: Portfolio environment
        strategy_name: Name of baseline strategy
        execution_strategy: Optional execution strategy (default: Sequential)

    Returns:
        Portfolio history DataFrame
    """
    registry = BaselineStrategyRegistry()
    strategy = registry.get(strategy_name)

    exec_strategy = execution_strategy or SequentialBacktestStrategy()
    return exec_strategy.execute_baseline(strategy, env)
