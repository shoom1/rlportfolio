"""
Abstract base class and concrete implementations for experiment tracking strategies.

This module implements the Strategy pattern for experiment tracking, allowing easy
switching between different tracking backends (W&B, MLflow, or no tracking).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import numpy as np
import pandas as pd


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking strategies.

    Implements the Strategy pattern for pluggable experiment tracking backends.
    Each concrete strategy handles initialization, logging, and cleanup for a
    specific tracking system (W&B, MLflow, etc.).
    """

    def __init__(self, experiment_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the experiment tracker.

        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary to log
        """
        self.experiment_name = experiment_name
        self.config = config or {}
        self.is_initialized = False

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the tracking backend.

        This is called when entering the context manager or explicitly starting tracking.

        Args:
            **kwargs: Backend-specific initialization parameters
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to the tracking backend.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/timestep number
        """
        pass

    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters/configuration.

        Args:
            params: Dictionary of hyperparameters
        """
        pass

    @abstractmethod
    def log_artifact(self, artifact_path: str, artifact_type: Optional[str] = None) -> None:
        """
        Log an artifact (file, model, etc.).

        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (model, dataset, etc.)
        """
        pass

    @abstractmethod
    def log_figure(self, figure_name: str, figure: Any, step: Optional[int] = None) -> None:
        """
        Log a matplotlib figure or plot.

        Args:
            figure_name: Name for the figure
            figure: Figure object (matplotlib.figure.Figure or similar)
            step: Optional step number
        """
        pass

    @abstractmethod
    def log_table(self, table_name: str, dataframe: pd.DataFrame) -> None:
        """
        Log a table/dataframe.

        Args:
            table_name: Name for the table
            dataframe: pandas DataFrame to log
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """
        Clean up and finish the tracking session.

        This is called when exiting the context manager or explicitly ending tracking.
        """
        pass

    @contextmanager
    def track(self, **init_kwargs):
        """
        Context manager for tracking experiments.

        Usage:
            with tracker.track(project="my_project"):
                # Training code here
                tracker.log_metrics({"loss": 0.5})

        Args:
            **init_kwargs: Backend-specific initialization parameters
        """
        try:
            self.initialize(**init_kwargs)
            yield self
        finally:
            self.finish()

    def log_portfolio_metrics(
        self,
        portfolio_value: float,
        weights: np.ndarray,
        cash_ratio: float,
        tickers: List[str],
        step: Optional[int] = None
    ) -> None:
        """
        Convenience method for logging portfolio-specific metrics.

        Args:
            portfolio_value: Current portfolio value
            weights: Array of asset weights
            cash_ratio: Cash as fraction of portfolio
            tickers: List of ticker symbols
            step: Optional step number
        """
        metrics = {
            'portfolio/value': portfolio_value,
            'portfolio/cash_ratio': cash_ratio,
        }

        # Add individual asset weights
        for ticker, weight in zip(tickers, weights):
            metrics[f'portfolio/weight_{ticker}'] = float(weight)

        self.log_metrics(metrics, step=step)

    def log_episode_summary(
        self,
        episode_reward: float,
        episode_length: int,
        final_portfolio_value: float,
        sharpe_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        step: Optional[int] = None
    ) -> None:
        """
        Convenience method for logging episode summary metrics.

        Args:
            episode_reward: Total episode reward
            episode_length: Length of episode
            final_portfolio_value: Final portfolio value
            sharpe_ratio: Sharpe ratio (if available)
            max_drawdown: Maximum drawdown (if available)
            step: Optional step number
        """
        metrics = {
            'episode/reward': episode_reward,
            'episode/length': episode_length,
            'episode/final_value': final_portfolio_value,
        }

        if sharpe_ratio is not None:
            metrics['episode/sharpe_ratio'] = sharpe_ratio

        if max_drawdown is not None:
            metrics['episode/max_drawdown'] = max_drawdown

        self.log_metrics(metrics, step=step)


class MockTracker(ExperimentTracker):
    """
    Mock tracker that doesn't log anything.

    Useful for testing or when you want to disable tracking without
    changing your code.
    """

    def initialize(self, **kwargs) -> None:
        """Initialize (no-op)."""
        self.is_initialized = True

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics (no-op)."""
        pass

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters (no-op)."""
        pass

    def log_artifact(self, artifact_path: str, artifact_type: Optional[str] = None) -> None:
        """Log artifact (no-op)."""
        pass

    def log_figure(self, figure_name: str, figure: Any, step: Optional[int] = None) -> None:
        """Log figure (no-op)."""
        pass

    def log_table(self, table_name: str, dataframe: pd.DataFrame) -> None:
        """Log table (no-op)."""
        pass

    def finish(self) -> None:
        """Finish (no-op)."""
        self.is_initialized = False


class VerboseMockTracker(ExperimentTracker):
    """
    Mock tracker that prints what it would log (for debugging).

    Useful for testing the tracking flow without actually sending data
    to a tracking backend.
    """

    def __init__(self, experiment_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(experiment_name, config)
        self.logged_metrics = []
        self.logged_params = {}
        self.logged_artifacts = []

    def initialize(self, **kwargs) -> None:
        """Initialize with verbose output."""
        self.is_initialized = True
        print(f"[VerboseMockTracker] Initialized experiment: {self.experiment_name}")
        if kwargs:
            print(f"[VerboseMockTracker] Init kwargs: {kwargs}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics with verbose output."""
        self.logged_metrics.append((step, metrics))
        step_str = f"step={step}" if step is not None else "no step"
        print(f"[VerboseMockTracker] Logging metrics ({step_str}): {metrics}")

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters with verbose output."""
        self.logged_params.update(params)
        print(f"[VerboseMockTracker] Logging hyperparameters: {params}")

    def log_artifact(self, artifact_path: str, artifact_type: Optional[str] = None) -> None:
        """Log artifact with verbose output."""
        self.logged_artifacts.append((artifact_path, artifact_type))
        type_str = f"type={artifact_type}" if artifact_type else "no type"
        print(f"[VerboseMockTracker] Logging artifact ({type_str}): {artifact_path}")

    def log_figure(self, figure_name: str, figure: Any, step: Optional[int] = None) -> None:
        """Log figure with verbose output."""
        step_str = f"step={step}" if step is not None else "no step"
        print(f"[VerboseMockTracker] Logging figure ({step_str}): {figure_name}")

    def log_table(self, table_name: str, dataframe: pd.DataFrame) -> None:
        """Log table with verbose output."""
        print(f"[VerboseMockTracker] Logging table: {table_name} (shape={dataframe.shape})")

    def finish(self) -> None:
        """Finish with verbose output."""
        print(f"[VerboseMockTracker] Finished experiment: {self.experiment_name}")
        print(f"[VerboseMockTracker] Total metrics logged: {len(self.logged_metrics)}")
        print(f"[VerboseMockTracker] Total artifacts logged: {len(self.logged_artifacts)}")
        self.is_initialized = False
