"""
Abstract base class and concrete implementations for experiment tracking strategies.

This module implements the Strategy pattern for experiment tracking with a
Template Method skeleton so subclasses only implement backend-specific hooks
(`_initialize_backend`, `_log_metrics_impl`, ...). The public API — `initialize`,
`log_metrics`, `finish`, etc. — is shared in this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import numpy as np
import pandas as pd


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking strategies.

    Subclasses override the `_*_impl` / `_initialize_backend` / `_finish_backend`
    hooks. The public methods on this class enforce the initialization
    invariant and flip the is_initialized flag uniformly.
    """

    def __init__(self, experiment_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.experiment_name = experiment_name
        self.config = config or {}
        self.is_initialized = False

    # ------------------------------------------------------------------
    # Public API — concrete, shared across backends
    # ------------------------------------------------------------------

    def initialize(self, **kwargs) -> None:
        """Initialize the backend and mark the tracker ready."""
        self._initialize_backend(**kwargs)
        self.is_initialized = True

    def finish(self) -> None:
        """Clean up the backend. Idempotent."""
        if not self.is_initialized:
            return
        self._finish_backend()
        self.is_initialized = False

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self._require_initialized()
        self._log_metrics_impl(metrics, step)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        self._require_initialized()
        self._log_hyperparameters_impl(params)

    def log_artifact(self, artifact_path: str, artifact_type: Optional[str] = None) -> None:
        self._require_initialized()
        self._log_artifact_impl(artifact_path, artifact_type)

    def log_figure(self, figure_name: str, figure: Any, step: Optional[int] = None) -> None:
        self._require_initialized()
        self._log_figure_impl(figure_name, figure, step)

    def log_table(self, table_name: str, dataframe: pd.DataFrame) -> None:
        self._require_initialized()
        self._log_table_impl(table_name, dataframe)

    @contextmanager
    def track(self, **init_kwargs):
        """Context manager that initialises on enter and cleans up on exit."""
        try:
            self.initialize(**init_kwargs)
            yield self
        finally:
            self.finish()

    def _require_initialized(self) -> None:
        if not self.is_initialized:
            raise RuntimeError(
                f"{type(self).__name__} not initialized. "
                f"Call initialize() or use the track() context manager."
            )

    # ------------------------------------------------------------------
    # Abstract backend hooks — override in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _initialize_backend(self, **kwargs) -> None:
        """Open connections, start a run, etc."""

    @abstractmethod
    def _finish_backend(self) -> None:
        """Flush buffers and close the run."""

    @abstractmethod
    def _log_metrics_impl(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def _log_hyperparameters_impl(self, params: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def _log_artifact_impl(self, artifact_path: str, artifact_type: Optional[str] = None) -> None:
        ...

    @abstractmethod
    def _log_figure_impl(self, figure_name: str, figure: Any, step: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def _log_table_impl(self, table_name: str, dataframe: pd.DataFrame) -> None:
        ...

    # ------------------------------------------------------------------
    # Convenience helpers shared across backends
    # ------------------------------------------------------------------

    def log_portfolio_metrics(
        self,
        portfolio_value: float,
        weights: np.ndarray,
        cash_ratio: float,
        tickers: List[str],
        step: Optional[int] = None
    ) -> None:
        """Log portfolio-specific metrics in a standardised namespace."""
        metrics = {
            'portfolio/value': portfolio_value,
            'portfolio/cash_ratio': cash_ratio,
        }
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
        """Log end-of-episode summary metrics."""
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
    """Tracker that swallows every call. Useful for disabling tracking."""

    def _initialize_backend(self, **kwargs) -> None:
        pass

    def _finish_backend(self) -> None:
        pass

    def _log_metrics_impl(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def _log_hyperparameters_impl(self, params: Dict[str, Any]) -> None:
        pass

    def _log_artifact_impl(self, artifact_path: str, artifact_type: Optional[str] = None) -> None:
        pass

    def _log_figure_impl(self, figure_name: str, figure: Any, step: Optional[int] = None) -> None:
        pass

    def _log_table_impl(self, table_name: str, dataframe: pd.DataFrame) -> None:
        pass


class VerboseMockTracker(ExperimentTracker):
    """Mock tracker that prints every call. Useful for debugging the flow."""

    def __init__(self, experiment_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(experiment_name, config)
        self.logged_metrics: list = []
        self.logged_params: dict = {}
        self.logged_artifacts: list = []

    def _initialize_backend(self, **kwargs) -> None:
        print(f"[VerboseMockTracker] Initialized experiment: {self.experiment_name}")
        if kwargs:
            print(f"[VerboseMockTracker] Init kwargs: {kwargs}")

    def _finish_backend(self) -> None:
        print(f"[VerboseMockTracker] Finished experiment: {self.experiment_name}")
        print(f"[VerboseMockTracker] Total metrics logged: {len(self.logged_metrics)}")
        print(f"[VerboseMockTracker] Total artifacts logged: {len(self.logged_artifacts)}")

    def _log_metrics_impl(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        self.logged_metrics.append((step, metrics))
        step_str = f"step={step}" if step is not None else "no step"
        print(f"[VerboseMockTracker] Logging metrics ({step_str}): {metrics}")

    def _log_hyperparameters_impl(self, params: Dict[str, Any]) -> None:
        self.logged_params.update(params)
        print(f"[VerboseMockTracker] Logging hyperparameters: {params}")

    def _log_artifact_impl(self, artifact_path: str, artifact_type: Optional[str] = None) -> None:
        self.logged_artifacts.append((artifact_path, artifact_type))
        type_str = f"type={artifact_type}" if artifact_type else "no type"
        print(f"[VerboseMockTracker] Logging artifact ({type_str}): {artifact_path}")

    def _log_figure_impl(self, figure_name: str, figure: Any, step: Optional[int] = None) -> None:
        step_str = f"step={step}" if step is not None else "no step"
        print(f"[VerboseMockTracker] Logging figure ({step_str}): {figure_name}")

    def _log_table_impl(self, table_name: str, dataframe: pd.DataFrame) -> None:
        print(f"[VerboseMockTracker] Logging table: {table_name} (shape={dataframe.shape})")
