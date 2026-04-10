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

    Integrates with W&B for experiment tracking, logging metrics, hyperparameters,
    artifacts, and visualizations.
    """

    def __init__(self, experiment_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize W&B tracker.

        Args:
            experiment_name: Name of the experiment (W&B run name)
            config: Configuration dictionary to log
        """
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Install it with: pip install wandb"
            )

        super().__init__(experiment_name, config)
        self.run = None

    def initialize(self, **kwargs) -> None:
        """
        Initialize W&B run.

        Args:
            **kwargs: W&B-specific parameters:
                - project: W&B project name
                - entity: W&B entity (username or team)
                - tags: List of tags for the run
                - notes: Notes for the run
                - group: Group name for the run
                - job_type: Job type (train, eval, etc.)
                - mode: "online", "offline", or "disabled"
                - Any other wandb.init() parameters
        """
        # Extract project (required for W&B)
        project = kwargs.pop('project', 'rl-portfolio')

        # Prepare init arguments
        init_args = {
            'project': project,
            'name': self.experiment_name,
            'config': self.config,
            **kwargs
        }

        # Initialize W&B run
        self.run = wandb.init(**init_args)
        self.is_initialized = True

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/timestep number
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        wandb.log(metrics, step=step)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters to W&B config.

        Args:
            params: Dictionary of hyperparameters
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        wandb.config.update(params)

    def log_artifact(self, artifact_path: str, artifact_type: Optional[str] = None) -> None:
        """
        Log an artifact to W&B.

        Args:
            artifact_path: Path to the artifact file
            artifact_type: Type of artifact (model, dataset, etc.)
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        artifact_type = artifact_type or "file"
        artifact = wandb.Artifact(
            name=f"{self.experiment_name}_{artifact_type}",
            type=artifact_type
        )
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def log_figure(self, figure_name: str, figure: Any, step: Optional[int] = None) -> None:
        """
        Log a matplotlib figure to W&B.

        Args:
            figure_name: Name for the figure
            figure: matplotlib.figure.Figure object
            step: Optional step number
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        wandb.log({figure_name: wandb.Image(figure)}, step=step)

    def log_table(self, table_name: str, dataframe: pd.DataFrame) -> None:
        """
        Log a table to W&B.

        Args:
            table_name: Name for the table
            dataframe: pandas DataFrame to log
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        table = wandb.Table(dataframe=dataframe)
        wandb.log({table_name: table})

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.is_initialized and self.run is not None:
            wandb.finish()
            self.run = None
            self.is_initialized = False

    def watch_model(self, model: Any, log_freq: int = 1000, log_graph: bool = True) -> None:
        """
        Watch a model for gradient and parameter tracking.

        Args:
            model: PyTorch or Keras model
            log_freq: Frequency of logging
            log_graph: Whether to log the computational graph
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        wandb.watch(model, log_freq=log_freq, log_graph=log_graph)

    def save_model(self, model_path: str, aliases: Optional[list] = None) -> None:
        """
        Save and upload a model as a W&B artifact.

        Args:
            model_path: Path to the saved model
            aliases: List of aliases for the artifact (e.g., ["latest", "best"])
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        aliases = aliases or ["latest"]
        artifact = wandb.Artifact(
            name=f"{self.experiment_name}_model",
            type="model"
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact, aliases=aliases)
