"""
Callback class for integrating experiment trackers into Stable-Baselines3 training.
"""

from typing import Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from .experiment_tracker import ExperimentTracker


class ExperimentTrackingCallback(BaseCallback):
    """
    Callback for logging metrics to an experiment tracker during training.

    Integrates any ExperimentTracker implementation (W&B, MLflow, Mock, etc.)
    into the Stable-Baselines3 training loop using the callback pattern.

    Usage:
        tracker = WandbTracker("my_experiment", config={...})
        callback = ExperimentTrackingCallback(tracker, log_freq=1000)

        with tracker.track(project="rl-portfolio"):
            model.learn(total_timesteps=100000, callback=callback)
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        log_freq: int = 1000,
        log_portfolio_weights: bool = True,
        log_rollout_stats: bool = True,
        verbose: int = 0
    ):
        """
        Initialize the tracking callback.

        Args:
            tracker: ExperimentTracker instance to use for logging
            log_freq: Frequency of logging (in timesteps)
            log_portfolio_weights: Whether to log individual portfolio weights
            log_rollout_stats: Whether to log rollout statistics
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.tracker = tracker
        self.log_freq = log_freq
        self.log_portfolio_weights = log_portfolio_weights
        self.log_rollout_stats = log_rollout_stats

        # Track episode metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_portfolio_values = []

    def _on_training_start(self) -> None:
        """Called when training starts."""
        # Log model hyperparameters
        if hasattr(self.model, 'get_parameters'):
            try:
                params = self.model.get_parameters()
                self.tracker.log_hyperparameters(params)
            except Exception:
                pass  # Skip if parameters can't be extracted

    def _on_step(self) -> bool:
        """
        Called at each environment step.

        Returns:
            True to continue training, False to stop
        """
        # Log at specified frequency
        if self.n_calls % self.log_freq == 0:
            self._log_step_metrics()

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (for on-policy algorithms)."""
        if self.log_rollout_stats:
            self._log_rollout_metrics()

    def _log_step_metrics(self) -> None:
        """Log metrics at the current step."""
        metrics = {}

        # Log basic training metrics
        metrics['train/timesteps'] = self.num_timesteps

        # Log episode info if available
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            metrics['episode/reward'] = ep_info.get('r', 0)
            metrics['episode/length'] = ep_info.get('l', 0)

        # Log portfolio-specific metrics if environment supports it
        try:
            env = self.training_env.envs[0].unwrapped

            if hasattr(env, 'portfolio_value'):
                metrics['portfolio/value'] = env.portfolio_value

            if hasattr(env, 'cash') and hasattr(env, 'portfolio_value'):
                cash_ratio = env.cash / env.portfolio_value if env.portfolio_value > 0 else 0
                metrics['portfolio/cash_ratio'] = cash_ratio

            # Log individual portfolio weights
            if self.log_portfolio_weights and hasattr(env, 'weights') and hasattr(env, 'tickers'):
                for ticker, weight in zip(env.tickers, env.weights):
                    metrics[f'portfolio/weight_{ticker}'] = float(weight)

        except (AttributeError, IndexError):
            pass  # Skip portfolio metrics if not available

        # Log learning rate if available
        if hasattr(self.model, 'learning_rate'):
            try:
                lr = self.model.learning_rate
                if callable(lr):
                    lr = lr(1.0)  # Get current LR
                metrics['train/learning_rate'] = float(lr)
            except Exception:
                pass

        # Log to tracker
        if metrics:
            self.tracker.log_metrics(metrics, step=self.num_timesteps)

    def _log_rollout_metrics(self) -> None:
        """Log rollout statistics (for on-policy algorithms like PPO)."""
        try:
            # Check if model has rollout buffer (on-policy algorithms)
            if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer.size() > 0:
                buffer = self.model.rollout_buffer

                # Log reward statistics
                rewards = buffer.rewards.flatten()
                metrics = {
                    'rollout/mean_reward': float(np.mean(rewards)),
                    'rollout/std_reward': float(np.std(rewards)),
                    'rollout/min_reward': float(np.min(rewards)),
                    'rollout/max_reward': float(np.max(rewards)),
                }

                # Log value estimates if available
                if hasattr(buffer, 'values'):
                    values = buffer.values.flatten()
                    metrics.update({
                        'rollout/mean_value': float(np.mean(values)),
                        'rollout/std_value': float(np.std(values)),
                    })

                # Log advantages if available
                if hasattr(buffer, 'advantages'):
                    advantages = buffer.advantages.flatten()
                    metrics.update({
                        'rollout/mean_advantage': float(np.mean(advantages)),
                        'rollout/std_advantage': float(np.std(advantages)),
                    })

                self.tracker.log_metrics(metrics, step=self.num_timesteps)

        except Exception as e:
            if self.verbose >= 1:
                print(f"Warning: Could not log rollout metrics: {e}")

    def _on_training_end(self) -> None:
        """Called when training ends."""
        # Log final metrics
        final_metrics = {
            'train/total_timesteps': self.num_timesteps,
        }

        try:
            env = self.training_env.envs[0].unwrapped
            if hasattr(env, 'portfolio_value'):
                final_metrics['final/portfolio_value'] = env.portfolio_value
        except (AttributeError, IndexError):
            pass

        self.tracker.log_metrics(final_metrics, step=self.num_timesteps)


class MultiTrackerCallback(BaseCallback):
    """
    Callback that logs to multiple experiment trackers simultaneously.

    Useful when you want to log to both W&B and MLflow, for example.

    Usage:
        wandb_tracker = WandbTracker("my_experiment")
        mlflow_tracker = MLflowTracker("my_experiment")
        callback = MultiTrackerCallback([wandb_tracker, mlflow_tracker])

        with wandb_tracker.track(project="rl-portfolio"):
            with mlflow_tracker.track():
                model.learn(total_timesteps=100000, callback=callback)
    """

    def __init__(
        self,
        trackers: list[ExperimentTracker],
        log_freq: int = 1000,
        log_portfolio_weights: bool = True,
        log_rollout_stats: bool = True,
        verbose: int = 0
    ):
        """
        Initialize multi-tracker callback.

        Args:
            trackers: List of ExperimentTracker instances
            log_freq: Frequency of logging (in timesteps)
            log_portfolio_weights: Whether to log individual portfolio weights
            log_rollout_stats: Whether to log rollout statistics
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.callbacks = [
            ExperimentTrackingCallback(
                tracker,
                log_freq=log_freq,
                log_portfolio_weights=log_portfolio_weights,
                log_rollout_stats=log_rollout_stats,
                verbose=verbose
            )
            for tracker in trackers
        ]

    def init_callback(self, model):
        """Initialize all callbacks."""
        for callback in self.callbacks:
            callback.init_callback(model)

    def _on_training_start(self) -> None:
        """Called when training starts."""
        for callback in self.callbacks:
            callback._on_training_start()

    def _on_step(self) -> bool:
        """Called at each environment step."""
        for callback in self.callbacks:
            if not callback._on_step():
                return False
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        for callback in self.callbacks:
            callback._on_rollout_end()

    def _on_training_end(self) -> None:
        """Called when training ends."""
        for callback in self.callbacks:
            callback._on_training_end()
