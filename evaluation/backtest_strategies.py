"""
Backtest execution strategies using the Strategy pattern.

Provides different execution modes for portfolio backtesting:
- Sequential: Standard step-by-step execution
- Vectorized: Fast execution for simple strategies (future enhancement)
- WalkForward: Walk-forward analysis with rolling windows
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BacktestStrategy(ABC):
    """
    Abstract base class for backtest execution strategies.

    Different strategies implement different execution modes
    (sequential, vectorized, walk-forward, etc.)
    """

    @abstractmethod
    def execute_agent(self, agent, env, deterministic: bool = True, **kwargs) -> pd.DataFrame:
        """
        Execute backtest using a trained agent.

        Args:
            agent: Trained SB3 agent
            env: Portfolio environment
            deterministic: Whether to use deterministic actions
            **kwargs: Additional strategy-specific parameters

        Returns:
            Portfolio history DataFrame
        """
        pass

    @abstractmethod
    def execute_baseline(self, strategy, env, **kwargs) -> pd.DataFrame:
        """
        Execute backtest using a baseline strategy.

        Args:
            strategy: BaselineStrategy instance
            env: Portfolio environment
            **kwargs: Additional strategy-specific parameters

        Returns:
            Portfolio history DataFrame
        """
        pass


class SequentialBacktestStrategy(BacktestStrategy):
    """
    Standard sequential backtesting.

    Executes environment step-by-step from start to end.
    This is the default and most common execution mode.
    """

    def execute_agent(self, agent, env, deterministic: bool = True, **kwargs) -> pd.DataFrame:
        """
        Execute agent backtest sequentially.

        Args:
            agent: Trained SB3 agent
            env: Portfolio environment
            deterministic: Whether to use deterministic actions

        Returns:
            Portfolio history DataFrame
        """
        obs, info = env.reset()
        done = False

        while not done:
            action, _states = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        return env.get_portfolio_history()

    def execute_baseline(self, strategy, env, **kwargs) -> pd.DataFrame:
        """
        Execute baseline strategy backtest sequentially.

        Args:
            strategy: BaselineStrategy instance
            env: Portfolio environment

        Returns:
            Portfolio history DataFrame
        """
        strategy.reset()

        obs, info = env.reset()
        done = False
        step = 0

        while not done:
            action = strategy.get_action(env, step)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        return env.get_portfolio_history()


class WalkForwardBacktestStrategy(BacktestStrategy):
    """
    Walk-forward analysis backtesting.

    Splits data into rolling train/test windows and executes multiple backtests.
    Useful for assessing strategy robustness over different market regimes.
    """

    def __init__(self, train_window: int, test_window: int, step_size: Optional[int] = None):
        """
        Initialize walk-forward strategy.

        Args:
            train_window: Number of periods for training window
            test_window: Number of periods for testing window
            step_size: Step size for rolling window (default: test_window)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size or test_window
        self.fold_results = []  # Store results for each fold

    def execute_agent(self, agent, env, deterministic: bool = True, **kwargs) -> pd.DataFrame:
        """
        Execute walk-forward backtest for agent.

        Note: For agents, we don't retrain - just evaluate on rolling windows.
        For true walk-forward with retraining, use a dedicated training pipeline.

        Args:
            agent: Trained SB3 agent
            env: Portfolio environment
            deterministic: Whether to use deterministic actions

        Returns:
            Aggregated portfolio history DataFrame from all test windows
        """
        self.fold_results = []

        total_steps = env.n_steps

        histories = []
        start_idx = 0

        while start_idx + self.train_window + self.test_window <= total_steps:
            test_start = start_idx + self.train_window
            test_end = test_start + self.test_window

            obs, info = env.reset(start_step=test_start)

            done = False
            while not done and env.current_step < test_end:
                action, _states = agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            fold_history = env.get_portfolio_history()
            histories.append(fold_history)

            self.fold_results.append({
                'train_start': start_idx,
                'train_end': test_start,
                'test_start': test_start,
                'test_end': test_end,
                'n_steps': len(fold_history)
            })

            start_idx += self.step_size

        if histories:
            combined_history = pd.concat(histories, ignore_index=True)
            return combined_history
        else:
            raise ValueError("Not enough data for walk-forward analysis")

    def execute_baseline(self, strategy, env, **kwargs) -> pd.DataFrame:
        """
        Execute walk-forward backtest for baseline strategy.

        Args:
            strategy: BaselineStrategy instance
            env: Portfolio environment

        Returns:
            Aggregated portfolio history DataFrame from all test windows
        """
        self.fold_results = []

        total_steps = env.n_steps
        histories = []
        start_idx = 0

        while start_idx + self.train_window + self.test_window <= total_steps:
            test_start = start_idx + self.train_window
            test_end = test_start + self.test_window

            strategy.reset()
            obs, info = env.reset(start_step=test_start)

            done = False
            step = 0
            while not done and env.current_step < test_end:
                action = strategy.get_action(env, step)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

            fold_history = env.get_portfolio_history()
            histories.append(fold_history)

            self.fold_results.append({
                'train_start': start_idx,
                'train_end': test_start,
                'test_start': test_start,
                'test_end': test_end,
                'n_steps': len(fold_history)
            })

            start_idx += self.step_size

        if histories:
            combined_history = pd.concat(histories, ignore_index=True)
            return combined_history
        else:
            raise ValueError("Not enough data for walk-forward analysis")

    def get_fold_results(self) -> list:
        """Get metadata about each fold in walk-forward analysis."""
        return self.fold_results


class MonteCarloBacktestStrategy(BacktestStrategy):
    """
    Monte Carlo backtesting with random seeds.

    Runs multiple backtests with different random seeds to assess
    strategy stability and get confidence intervals on performance.
    """

    def __init__(self, n_simulations: int = 100, seeds: Optional[list] = None):
        """
        Initialize Monte Carlo strategy.

        Args:
            n_simulations: Number of Monte Carlo simulations
            seeds: Optional list of random seeds (generated if not provided).
                Must have at least n_simulations entries when supplied.
        """
        self.n_simulations = n_simulations
        self.seeds = seeds if seeds is not None else list(range(n_simulations))
        if len(self.seeds) < n_simulations:
            raise ValueError(
                f"seeds has {len(self.seeds)} entries but n_simulations={n_simulations}"
            )
        self.simulation_results = []  # Store all simulation results

    def execute_agent(self, agent, env, deterministic: bool = False, **kwargs) -> pd.DataFrame:
        """
        Execute Monte Carlo backtest for agent.

        Args:
            agent: Trained SB3 agent
            env: Portfolio environment
            deterministic: Whether to use deterministic actions (typically False for MC)

        Returns:
            Mean portfolio history across all simulations
        """
        self.simulation_results = []
        all_histories = []

        for seed in self.seeds[:self.n_simulations]:
            # Set random seed
            np.random.seed(seed)

            # Run single backtest
            obs, info = env.reset(seed=seed)
            done = False

            while not done:
                action, _states = agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            history = env.get_portfolio_history()
            all_histories.append(history)

            # Store simulation result
            self.simulation_results.append({
                'seed': seed,
                'final_value': history['value'].iloc[-1],
                'total_return': (history['value'].iloc[-1] / history['value'].iloc[0]) - 1
            })

        # Compute mean history across simulations
        # Note: assumes all histories have same length
        mean_history = self._aggregate_histories(all_histories, method='mean')
        return mean_history

    def execute_baseline(self, strategy, env, **kwargs) -> pd.DataFrame:
        """
        Execute Monte Carlo backtest for baseline strategy.

        Args:
            strategy: BaselineStrategy instance
            env: Portfolio environment

        Returns:
            Mean portfolio history across all simulations
        """
        self.simulation_results = []
        all_histories = []

        for seed in self.seeds[:self.n_simulations]:
            np.random.seed(seed)

            strategy.reset(seed=seed)
            obs, info = env.reset(seed=seed)
            done = False
            step = 0

            while not done:
                action = strategy.get_action(env, step)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

            history = env.get_portfolio_history()
            all_histories.append(history)

            self.simulation_results.append({
                'seed': seed,
                'final_value': history['value'].iloc[-1],
                'total_return': (history['value'].iloc[-1] / history['value'].iloc[0]) - 1
            })

        mean_history = self._aggregate_histories(all_histories, method='mean')
        return mean_history

    def _aggregate_histories(self, histories: list, method: str = 'mean') -> pd.DataFrame:
        """
        Aggregate multiple portfolio histories.

        Truncates to the shortest history to handle ragged runs. Only value,
        return, and date are aggregated — run-dependent columns (weights, cash,
        transaction_cost) are dropped because their "mean" across simulations
        is not meaningful.

        Args:
            histories: List of portfolio history DataFrames
            method: Aggregation method ('mean', 'median')

        Returns:
            Aggregated portfolio history with columns: date, value, return
        """
        if not histories:
            raise ValueError("No histories to aggregate")

        min_len = min(len(h) for h in histories)
        all_values = np.array([h['value'].values[:min_len] for h in histories])

        if method == 'mean':
            agg_values = np.mean(all_values, axis=0)
        elif method == 'median':
            agg_values = np.median(all_values, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        agg_returns = np.concatenate([[0], np.diff(agg_values) / agg_values[:-1]])

        agg_history = pd.DataFrame({
            'value': agg_values,
            'return': agg_returns,
        })
        if 'date' in histories[0].columns:
            agg_history['date'] = histories[0]['date'].values[:min_len]

        return agg_history

    def get_simulation_results(self) -> list:
        """Get results from all Monte Carlo simulations."""
        return self.simulation_results

    def get_confidence_intervals(self, confidence: float = 0.95) -> Dict[str, tuple]:
        """
        Calculate confidence intervals for final value and total return.

        Args:
            confidence: Confidence level (default: 0.95 for 95% CI)

        Returns:
            Dictionary with confidence intervals
        """
        if not self.simulation_results:
            raise ValueError("No simulation results available")

        alpha = 1 - confidence
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        final_values = [r['final_value'] for r in self.simulation_results]
        total_returns = [r['total_return'] for r in self.simulation_results]

        return {
            'final_value_ci': (
                np.percentile(final_values, lower_pct),
                np.percentile(final_values, upper_pct)
            ),
            'total_return_ci': (
                np.percentile(total_returns, lower_pct),
                np.percentile(total_returns, upper_pct)
            )
        }
