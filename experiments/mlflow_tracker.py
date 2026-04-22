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

    Implements backend hooks; public API, initialization guards, and
    idempotent finish are handled by ExperimentTracker.
    """

    def __init__(self, experiment_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "mlflow is not installed. Install it with: pip install mlflow"
            )
        super().__init__(experiment_name, config)
        self.run = None
        self.experiment_id = None

    def _initialize_backend(self, **kwargs) -> None:
        """Start an MLflow run.

        Recognized kwargs:
            tracking_uri, experiment_name, run_name, tags, description, nested.
        """
        tracking_uri = kwargs.pop('tracking_uri', None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment_name = kwargs.pop('experiment_name', self.experiment_name or 'rl-portfolio')
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            # Experiment already exists
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        mlflow.set_experiment(experiment_name)

        run_name = kwargs.pop('run_name', self.experiment_name)
        tags = kwargs.pop('tags', {})
        description = kwargs.pop('description', None)
        nested = kwargs.pop('nested', False)

        self.run = mlflow.start_run(
            run_name=run_name,
            tags=tags,
            description=description,
            nested=nested,
        )

        if self.config:
            mlflow.log_params(self._flatten_dict(self.config))

    def _finish_backend(self) -> None:
        if self.run is not None:
            mlflow.end_run()
            self.run = None

    def _log_metrics_impl(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)

    def _log_hyperparameters_impl(self, params: Dict[str, Any]) -> None:
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)

    def _log_artifact_impl(self, artifact_path: str, artifact_type: Optional[str] = None) -> None:
        path = Path(artifact_path)
        if path.is_file():
            mlflow.log_artifact(artifact_path)
        elif path.is_dir():
            mlflow.log_artifacts(artifact_path)

    def _log_figure_impl(self, figure_name: str, figure: Any, step: Optional[int] = None) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            figure.savefig(tmp.name, format='png', bbox_inches='tight')
            tmp_path = tmp.name

        mlflow.log_artifact(tmp_path, artifact_path=f"figures/{figure_name}.png")
        Path(tmp_path).unlink()

    def _log_table_impl(self, table_name: str, dataframe: pd.DataFrame) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            dataframe.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        mlflow.log_artifact(tmp_path, artifact_path=f"tables/{table_name}.csv")
        Path(tmp_path).unlink()

    # ------------------------------------------------------------------
    # MLflow-specific extensions
    # ------------------------------------------------------------------

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Log a model to MLflow.

        Detects PyTorch, TensorFlow, or sklearn. Fails with a clear error for
        anything else (the old silent pickle fallback was removed).
        """
        self._require_initialized()

        try:
            import torch
            if isinstance(model, torch.nn.Module):
                mlflow.pytorch.log_model(
                    model,
                    artifact_path,
                    registered_model_name=registered_model_name,
                    **kwargs,
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
                    **kwargs,
                )
                return
        except ImportError:
            pass

        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        except Exception as e:
            raise TypeError(
                f"log_model cannot serialize {type(model).__name__}. "
                f"Use log_artifact() for SB3 agent zipfiles, or pass agent.policy "
                f"for the PyTorch path."
            ) from e

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Set tags for the current run."""
        self._require_initialized()
        mlflow.set_tags(tags)

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
