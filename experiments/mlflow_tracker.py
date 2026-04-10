"""
MLflow experiment tracking strategy.
"""

from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path
from .base import ExperimentTracker

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowTracker(ExperimentTracker):
    """
    MLflow experiment tracking strategy.

    Integrates with MLflow for experiment tracking, logging metrics, hyperparameters,
    artifacts, and models.
    """

    def __init__(self, experiment_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            config: Configuration dictionary to log
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "mlflow is not installed. Install it with: pip install mlflow"
            )

        super().__init__(experiment_name, config)
        self.run = None
        self.experiment_id = None

    def initialize(self, **kwargs) -> None:
        """
        Initialize MLflow run.

        Args:
            **kwargs: MLflow-specific parameters:
                - tracking_uri: MLflow tracking server URI
                - experiment_name: MLflow experiment name (overrides self.experiment_name)
                - run_name: Name for this specific run
                - tags: Dictionary of tags
                - description: Run description
                - nested: Whether this is a nested run
        """
        # Set tracking URI if provided
        tracking_uri = kwargs.pop('tracking_uri', None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Get or create experiment
        experiment_name = kwargs.pop('experiment_name', self.experiment_name or 'rl-portfolio')
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            # Experiment already exists
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        mlflow.set_experiment(experiment_name)

        # Extract MLflow-specific parameters
        run_name = kwargs.pop('run_name', self.experiment_name)
        tags = kwargs.pop('tags', {})
        description = kwargs.pop('description', None)
        nested = kwargs.pop('nested', False)

        # Start MLflow run
        self.run = mlflow.start_run(
            run_name=run_name,
            tags=tags,
            description=description,
            nested=nested
        )

        # Log configuration as parameters
        if self.config:
            mlflow.log_params(self._flatten_dict(self.config))

        self.is_initialized = True

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/timestep number
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters to MLflow.

        Args:
            params: Dictionary of hyperparameters
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        # Flatten nested dictionaries and log
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)

    def log_artifact(self, artifact_path: str, artifact_type: Optional[str] = None) -> None:
        """
        Log an artifact to MLflow.

        Args:
            artifact_path: Path to the artifact file or directory
            artifact_type: Type of artifact (ignored for MLflow, kept for compatibility)
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        path = Path(artifact_path)
        if path.is_file():
            mlflow.log_artifact(artifact_path)
        elif path.is_dir():
            mlflow.log_artifacts(artifact_path)

    def log_figure(self, figure_name: str, figure: Any, step: Optional[int] = None) -> None:
        """
        Log a matplotlib figure to MLflow.

        Args:
            figure_name: Name for the figure
            figure: matplotlib.figure.Figure object
            step: Optional step number (used in filename if provided)
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        # Save figure temporarily and log as artifact
        import tempfile
        import matplotlib.pyplot as plt

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            figure.savefig(tmp.name, format='png', bbox_inches='tight')
            tmp_path = tmp.name

        # Log the figure
        mlflow.log_artifact(tmp_path, artifact_path=f"figures/{figure_name}.png")

        # Clean up
        Path(tmp_path).unlink()

    def log_table(self, table_name: str, dataframe: pd.DataFrame) -> None:
        """
        Log a table to MLflow.

        Args:
            table_name: Name for the table
            dataframe: pandas DataFrame to log
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        # Save as CSV and log as artifact
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            dataframe.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        mlflow.log_artifact(tmp_path, artifact_path=f"tables/{table_name}.csv")

        # Clean up
        Path(tmp_path).unlink()

    def finish(self) -> None:
        """Finish the MLflow run."""
        if self.is_initialized and self.run is not None:
            mlflow.end_run()
            self.run = None
            self.is_initialized = False

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log a model to MLflow.

        Args:
            model: Model object (PyTorch, TensorFlow, or sklearn)
            artifact_path: Path within the run's artifact directory
            registered_model_name: If provided, register the model with this name
            **kwargs: Additional arguments passed to the specific log_model function
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        # Try to detect model type and log appropriately
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                mlflow.pytorch.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    **kwargs
                )
                return
        except ImportError:
            pass

        try:
            import tensorflow as tf
            if isinstance(model, tf.keras.Model):
                mlflow.tensorflow.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    **kwargs
                )
                return
        except ImportError:
            pass

        # Fallback to sklearn
        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )
        except Exception as e:
            # If all else fails, try generic pickling
            import pickle
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
                pickle.dump(model, tmp)
                tmp_path = tmp.name

            mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
            Path(tmp_path).unlink()

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """
        Set tags for the current run.

        Args:
            tags: Dictionary of tags
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() or use context manager.")

        mlflow.set_tags(tags)

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
