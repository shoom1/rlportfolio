"""
Demonstration of new design patterns in backtesting and metrics.

Shows:
1. Composition pattern for metrics calculation
2. Strategy pattern for backtest execution (Sequential, Walk-Forward, Monte Carlo)
"""

import sys
sys.path.append('..')

import numpy as np
from data.fetcher import DataFetcher
from data.features import FeatureEngineer
from environment.portfolio_env import PortfolioEnv
from environment.rewards import SharpeReward
from evaluation import (
    # Metrics - Composition pattern
    PortfolioMetrics,
    ReturnMetrics,
    RiskMetrics,
    DrawdownMetrics,
    WinLossMetrics,
    # Backtesting
    Backtester,
    # Backtest Strategies - Strategy pattern
    SequentialBacktestStrategy,
    WalkForwardBacktestStrategy,
    MonteCarloBacktestStrategy
)
from evaluation.metrics import BaseMetrics


def create_environment():
    """Create a portfolio environment for backtesting."""
    print("Fetching data...")
    fetcher = DataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = fetcher.get_latest_data(tickers, days=500)

    print("Computing features...")
    engineer = FeatureEngineer()
    features_data = engineer.compute_features(data)
    env_data = engineer.prepare_for_environment(features_data)

    feature_cols = engineer.create_observation_columns(normalize=True)
    env = PortfolioEnv(
        data=env_data,
        feature_columns=feature_cols,
        tickers=tickers,
        initial_balance=10000.0,
        transaction_cost=0.001,
        reward_function=SharpeReward(window=30),
        window_size=20
    )

    return env


def example_1_metrics_composition():
    """Example 1: Using individual metric calculators with composition."""
    print("\n" + "="*80)
    print("Example 1: Metrics Composition Pattern")
    print("="*80)

    env = create_environment()

    # Run a simple backtest with equal weight strategy
    backtester = Backtester()
    backtester.run_baseline(env, 'equal_weight')
    history = backtester.get_result('equal_weight').history

    # Use individual metric calculators
    print("\n--- Using Individual Metric Calculators ---")

    values = history['value'].values
    returns = BaseMetrics._get_returns(history)  # Use BaseMetrics helper

    return_calc = ReturnMetrics()
    returns_metrics = return_calc.calculate(
        values=values,
        n_periods=len(history)
    )
    print(f"Return Metrics: {returns_metrics}")

    risk_calc = RiskMetrics(risk_free_rate=0.02)
    risk_metrics = risk_calc.calculate(returns)
    print(f"Risk Metrics: {risk_metrics}")

    drawdown_calc = DrawdownMetrics()
    drawdown_metrics = drawdown_calc.calculate(
        values=values,
        n_periods=len(history)
    )
    print(f"Drawdown Metrics: {drawdown_metrics}")

    # Use composed PortfolioMetrics (facade)
    print("\n--- Using Composed PortfolioMetrics Facade ---")
    metrics_calc = PortfolioMetrics(risk_free_rate=0.02)
    all_metrics = metrics_calc.calculate_all_metrics(history)
    metrics_calc.print_metrics(all_metrics)


def example_2_sequential_backtest():
    """Example 2: Sequential backtest strategy (default)."""
    print("\n" + "="*80)
    print("Example 2: Sequential Backtest Strategy (Default)")
    print("="*80)

    env = create_environment()

    # Create backtester with sequential strategy (default)
    backtester = Backtester(
        risk_free_rate=0.02,
        execution_strategy=SequentialBacktestStrategy()
    )

    # Run multiple baseline strategies
    print("\nRunning sequential backtests...")
    backtester.run_baseline(env, 'equal_weight')

    env_copy = create_environment()  # Fresh environment
    backtester.run_baseline(env_copy, 'momentum_20')

    env_copy2 = create_environment()  # Fresh environment
    backtester.run_baseline(env_copy2, 'buy_and_hold')

    # Compare results
    backtester.print_comparison()


def example_3_walk_forward_backtest():
    """Example 3: Walk-forward backtesting."""
    print("\n" + "="*80)
    print("Example 3: Walk-Forward Backtest Strategy")
    print("="*80)

    env = create_environment()

    # Create backtester with walk-forward strategy
    walk_forward_strategy = WalkForwardBacktestStrategy(
        train_window=200,  # 200 days for "training" reference
        test_window=60,    # 60 days for testing
        step_size=30       # Roll forward 30 days at a time
    )

    backtester = Backtester(
        risk_free_rate=0.02,
        execution_strategy=walk_forward_strategy
    )

    print("\nRunning walk-forward backtest...")
    print(f"  Train window: {walk_forward_strategy.train_window} days")
    print(f"  Test window: {walk_forward_strategy.test_window} days")
    print(f"  Step size: {walk_forward_strategy.step_size} days")

    backtester.run_baseline(env, 'equal_weight', result_name='equal_weight_wf')

    # Get walk-forward fold results
    fold_results = walk_forward_strategy.get_fold_results()
    print(f"\nCompleted {len(fold_results)} walk-forward folds:")
    for i, fold in enumerate(fold_results):
        print(f"  Fold {i+1}: Train [{fold['train_start']}-{fold['train_end']}], "
              f"Test [{fold['test_start']}-{fold['test_end']}], Steps: {fold['n_steps']}")

    # Compute metrics
    metrics = backtester.compute_metrics('equal_weight_wf')
    backtester.metrics_calculator.print_metrics(metrics)


def example_4_monte_carlo_backtest():
    """Example 4: Monte Carlo backtesting."""
    print("\n" + "="*80)
    print("Example 4: Monte Carlo Backtest Strategy")
    print("="*80)

    env = create_environment()

    # Create backtester with Monte Carlo strategy
    mc_strategy = MonteCarloBacktestStrategy(
        n_simulations=20,  # Run 20 simulations
        seeds=list(range(42, 62))  # Use seeds 42-61
    )

    backtester = Backtester(
        risk_free_rate=0.02,
        execution_strategy=mc_strategy
    )

    print(f"\nRunning Monte Carlo backtest with {mc_strategy.n_simulations} simulations...")
    print("(Using random baseline to show variance across runs)")

    backtester.run_baseline(env, 'random', result_name='random_mc')

    # Get simulation results
    sim_results = mc_strategy.get_simulation_results()
    print(f"\nSimulation Results (first 5):")
    for i, result in enumerate(sim_results[:5]):
        print(f"  Seed {result['seed']}: Final Value=${result['final_value']:.2f}, "
              f"Return={result['total_return']:.2%}")

    # Get confidence intervals
    ci = mc_strategy.get_confidence_intervals(confidence=0.95)
    print(f"\n95% Confidence Intervals:")
    print(f"  Final Value: ${ci['final_value_ci'][0]:.2f} - ${ci['final_value_ci'][1]:.2f}")
    print(f"  Total Return: {ci['total_return_ci'][0]:.2%} - {ci['total_return_ci'][1]:.2%}")

    # Compute metrics on mean result
    metrics = backtester.compute_metrics('random_mc')
    backtester.metrics_calculator.print_metrics(metrics)


def example_5_switching_strategies():
    """Example 5: Switching execution strategies dynamically."""
    print("\n" + "="*80)
    print("Example 5: Switching Execution Strategies Dynamically")
    print("="*80)

    # Create backtester with default sequential strategy
    backtester = Backtester(risk_free_rate=0.02)

    # Run sequential backtest
    print("\n1. Running with Sequential Strategy...")
    env1 = create_environment()
    backtester.run_baseline(env1, 'equal_weight', result_name='eq_sequential')

    # Switch to walk-forward strategy
    print("\n2. Switching to Walk-Forward Strategy...")
    backtester.set_execution_strategy(
        WalkForwardBacktestStrategy(train_window=200, test_window=50, step_size=50)
    )
    env2 = create_environment()
    backtester.run_baseline(env2, 'equal_weight', result_name='eq_walkforward')

    # Switch to Monte Carlo strategy
    print("\n3. Switching to Monte Carlo Strategy...")
    backtester.set_execution_strategy(
        MonteCarloBacktestStrategy(n_simulations=10)
    )
    env3 = create_environment()
    backtester.run_baseline(env3, 'momentum_20', result_name='momentum_mc')

    # Compare all results
    print("\n--- Comparison of All Strategies ---")
    backtester.print_comparison(['eq_sequential', 'eq_walkforward', 'momentum_mc'])


def example_6_custom_metrics_subset():
    """Example 6: Calculate only specific metric subsets."""
    print("\n" + "="*80)
    print("Example 6: Calculate Custom Metric Subsets")
    print("="*80)

    env = create_environment()
    backtester = Backtester()
    backtester.run_baseline(env, 'equal_weight')
    history = backtester.get_result('equal_weight').history

    # Use BaseMetrics helper for consistent returns calculation
    values = history['value'].values
    returns = BaseMetrics._get_returns(history)

    # Calculate only risk metrics
    print("\n--- Risk Metrics Only ---")
    risk_calc = RiskMetrics(risk_free_rate=0.02)
    risk_metrics = risk_calc.calculate(returns)
    for metric, value in risk_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Calculate only win/loss metrics
    print("\n--- Win/Loss Metrics Only ---")
    winloss_calc = WinLossMetrics()
    winloss_metrics = winloss_calc.calculate(returns)
    for metric, value in winloss_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Calculate custom combination
    print("\n--- Custom Combination: Returns + Risk ---")
    return_calc = ReturnMetrics()
    combined_metrics = {}
    combined_metrics.update(return_calc.calculate(values, len(history)))
    combined_metrics.update(risk_calc.calculate(returns))
    for metric, value in combined_metrics.items():
        print(f"  {metric}: {value:.4f}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("BACKTEST & METRICS DESIGN PATTERNS DEMONSTRATION")
    print("="*80)

    print("\nThis demo shows the new design patterns:")
    print("  1. Composition Pattern for Metrics")
    print("  2. Strategy Pattern for Backtest Execution")
    print()

    # Run examples
    example_1_metrics_composition()
    example_2_sequential_backtest()
    example_6_custom_metrics_subset()

    # These examples are more complex - uncomment to try them:
    # example_3_walk_forward_backtest()
    # example_4_monte_carlo_backtest()
    # example_5_switching_strategies()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
