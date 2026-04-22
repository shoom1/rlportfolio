"""
Weights & Biases experiment tracking strategy.
"""

from typing import Dict, Any, Optional
import pandas as pd
from .base import ExperimentTracker

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbTracker(ExperimentTracker):
    """
    Weights & Biases experiment tracking strategy.

    Implements backend hooks; public API, initialization guards, and
    idempotent finish are handled by ExperimentTracker.
    """

    def __init__(self, experiment_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Install it with: pip install wandb"
            )
        super().__init__(experiment_name, config)
        self.run = None

    def _initialize_backend(self, **kwargs) -> None:
        project = kwargs.pop('project', 'rl-portfolio')
        init_args = {
            'project': project,
            'name': self.experiment_name,
            'config': self.config,
            **kwargs,
        }
        self.run = wandb.init(**init_args)

    def _finish_backend(self) -> None:
        if self.run is not None:
            wandb.finish()
            self.run = None

    def _log_metrics_impl(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        wandb.log(metrics, step=step)

    def _log_hyperparameters_impl(self, params: Dict[str, Any]) -> None:
        wandb.config.update(params)

    def _log_artifact_impl(self, artifact_path: str, artifact_type: Optional[str] = None) -> None:
        artifact_type = artifact_type or "file"
        artifact = wandb.Artifact(
            name=f"{self.experiment_name}_{artifact_type}",
            type=artifact_type,
        )
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def _log_figure_impl(self, figure_name: str, figure: Any, step: Optional[int] = None) -> None:
        wandb.log({figure_name: wandb.Image(figure)}, step=step)

    def _log_table_impl(self, table_name: str, dataframe: pd.DataFrame) -> None:
        table = wandb.Table(dataframe=dataframe)
        wandb.log({table_name: table})

    # ------------------------------------------------------------------
    # W&B-specific extensions
    # ------------------------------------------------------------------

    def watch_model(self, model: Any, log_freq: int = 1000, log_graph: bool = True) -> None:
        """Watch a model for gradient and parameter tracking."""
        self._require_initialized()
        wandb.watch(model, log_freq=log_freq, log_graph=log_graph)

    def save_model(self, model_path: str, aliases: Optional[list] = None) -> None:
        """Save and upload a model as a W&B artifact."""
        self._require_initialized()
        aliases = aliases or ["latest"]
        artifact = wandb.Artifact(
            name=f"{self.experiment_name}_model",
            type="model",
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact, aliases=aliases)
