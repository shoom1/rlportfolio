"""
Custom portfolio experiment tracker for domain-specific metrics.
Works alongside W&B for comprehensive experiment tracking.
"""

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from environment.constants import DEFAULT_INITIAL_BALANCE, DEFAULT_TRANSACTION_COST


_EXPERIMENT_ID_RE = re.compile(r'^[A-Za-z0-9_\-]+$')


class PortfolioExperiment:
    """Container for portfolio-specific experiment data."""

    def __init__(
        self,
        name: str,
        experiment_id: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        wandb_run_id: Optional[str] = None
    ):
        """
        Initialize portfolio experiment.

        Args:
            name: Experiment name
            experiment_id: Unique ID (auto-generated if not provided)
            description: Experiment description
            tags: List of tags for categorization
            wandb_run_id: Associated W&B run ID
        """
        self.name = name
        self.experiment_id = experiment_id or self._generate_id()
        self.description = description
        self.tags = tags or []
        self.wandb_run_id = wandb_run_id
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

        # Portfolio-specific configuration
        self.portfolio_config = {
            'tickers': [],
            'initial_balance': DEFAULT_INITIAL_BALANCE,
            'transaction_cost': DEFAULT_TRANSACTION_COST,
            'reward_function': None
        }

        # Portfolio performance metrics
        self.portfolio_metrics = {
            'final_value': None,
            'total_return': None,
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'max_drawdown': None,
            'calmar_ratio': None,
            'win_rate': None,
            'profit_factor': None,
            'total_trades': None,
            'avg_trade_cost': None
        }

        # Time series data
        self.portfolio_history = None  # Will store as DataFrame
        self.weights_history = []

        # Training metrics
        self.training_metrics = {
            'total_timesteps': 0,
            'episodes': 0,
            'training_time': None,
            'final_reward': None,
            'avg_episode_reward': None
        }

        # Model info
        self.model_info = {
            'algorithm': None,
            'policy': None,
            'policy_architecture': None,
            'hyperparameters': {},
            'model_path': None,
            'best_model_path': None
        }

        # Comparison with baselines
        self.baseline_comparison = {}

    def _generate_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"portfolio_exp_{timestamp}"

    def update_portfolio_config(self, config: Dict[str, Any]):
        """Update portfolio configuration."""
        self.portfolio_config.update(config)
        self.updated_at = datetime.now().isoformat()

    def set_portfolio_metrics(self, metrics: Dict[str, float]):
        """Set final portfolio performance metrics."""
        self.portfolio_metrics.update(metrics)
        self.updated_at = datetime.now().isoformat()

    def set_portfolio_history(self, history: pd.DataFrame):
        """Store portfolio value history."""
        self.portfolio_history = history
        self.updated_at = datetime.now().isoformat()

    def add_baseline_comparison(self, baseline_name: str, metrics: Dict[str, float]):
        """Add comparison with baseline strategy."""
        self.baseline_comparison[baseline_name] = metrics
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'description': self.description,
            'tags': self.tags,
            'wandb_run_id': self.wandb_run_id,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'portfolio_config': self.portfolio_config,
            'portfolio_metrics': self.portfolio_metrics,
            'training_metrics': self.training_metrics,
            'model_info': self.model_info,
            'baseline_comparison': self.baseline_comparison
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioExperiment':
        """Create from dictionary."""
        exp = cls(
            name=data['name'],
            experiment_id=data['experiment_id'],
            description=data.get('description', ''),
            tags=data.get('tags', []),
            wandb_run_id=data.get('wandb_run_id')
        )

        exp.created_at = data.get('created_at', exp.created_at)
        exp.updated_at = data.get('updated_at', exp.updated_at)
        exp.portfolio_config = data.get('portfolio_config', exp.portfolio_config)
        exp.portfolio_metrics = data.get('portfolio_metrics', exp.portfolio_metrics)
        exp.training_metrics = data.get('training_metrics', exp.training_metrics)
        exp.model_info = data.get('model_info', exp.model_info)
        exp.baseline_comparison = data.get('baseline_comparison', {})

        return exp


class PortfolioExperimentTracker:
    """Tracks portfolio-specific experiment data using SQLite."""

    VALID_METRIC_COLUMNS = {
        'final_value', 'total_return', 'sharpe_ratio',
        'max_drawdown', 'total_timesteps',
    }

    def __init__(self, storage_dir: str = "experiments/portfolio"):
        """
        Initialize tracker.

        Args:
            storage_dir: Directory for storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # SQLite database for structured queries
        self.db_path = self.storage_dir / "experiments.db"
        self._init_database()

        # Current experiment
        self.current_experiment: Optional[PortfolioExperiment] = None

    def _safe_experiment_path(self, experiment_id: str, suffix: str) -> Path:
        """Build a path inside storage_dir, rejecting IDs that could escape it.

        Why: experiment_id flows in from user-facing CLI args and external
        caller lists; concatenating it directly into a filesystem path would
        allow `../../etc/passwd`-style reads or deletes.
        """
        if not isinstance(experiment_id, str) or not _EXPERIMENT_ID_RE.match(experiment_id):
            raise ValueError(
                f"Invalid experiment_id: {experiment_id!r}. "
                f"Must match {_EXPERIMENT_ID_RE.pattern}."
            )
        path = (self.storage_dir / f"{experiment_id}{suffix}").resolve()
        storage_root = self.storage_dir.resolve()
        if storage_root != path.parent and storage_root not in path.parents:
            raise ValueError(f"Path escapes storage_dir: {path}")
        return path

    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                wandb_run_id TEXT,
                created_at TEXT,
                updated_at TEXT,
                algorithm TEXT,
                status TEXT,
                final_value REAL,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_timesteps INTEGER
            )
        ''')

        # Tags table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                experiment_id TEXT,
                tag TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')

        # Metrics table for time series
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics_history (
                experiment_id TEXT,
                metric_name TEXT,
                value REAL,
                step INTEGER,
                timestamp TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')

        conn.commit()
        conn.close()

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        wandb_run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> PortfolioExperiment:
        """Create new portfolio experiment."""
        experiment = PortfolioExperiment(
            name=name,
            description=description,
            tags=tags,
            wandb_run_id=wandb_run_id
        )

        if config:
            experiment.update_portfolio_config(config)

        self.current_experiment = experiment
        self._save_experiment(experiment)

        return experiment

    def _save_experiment(self, experiment: PortfolioExperiment):
        """Save experiment to database and JSON."""
        # Save to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO experiments
            (experiment_id, name, description, wandb_run_id, created_at, updated_at,
             algorithm, final_value, total_return, sharpe_ratio, max_drawdown, total_timesteps)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment.experiment_id,
            experiment.name,
            experiment.description,
            experiment.wandb_run_id,
            experiment.created_at,
            experiment.updated_at,
            experiment.model_info.get('algorithm'),
            experiment.portfolio_metrics.get('final_value'),
            experiment.portfolio_metrics.get('total_return'),
            experiment.portfolio_metrics.get('sharpe_ratio'),
            experiment.portfolio_metrics.get('max_drawdown'),
            experiment.training_metrics.get('total_timesteps')
        ))

        # Save tags
        cursor.execute('DELETE FROM tags WHERE experiment_id = ?', (experiment.experiment_id,))
        for tag in experiment.tags:
            cursor.execute('INSERT INTO tags (experiment_id, tag) VALUES (?, ?)',
                         (experiment.experiment_id, tag))

        conn.commit()
        conn.close()

        # Save detailed JSON
        json_path = self._safe_experiment_path(experiment.experiment_id, ".json")
        with open(json_path, 'w') as f:
            json.dump(experiment.to_dict(), f, indent=2, default=str)

        # Save portfolio history if available
        if experiment.portfolio_history is not None:
            csv_path = self._safe_experiment_path(experiment.experiment_id, "_history.csv")
            experiment.portfolio_history.to_csv(csv_path)

    def load_experiment(self, experiment_id: str) -> PortfolioExperiment:
        """Load experiment from storage."""
        json_path = self._safe_experiment_path(experiment_id, ".json")

        if not json_path.exists():
            raise ValueError(f"Experiment {experiment_id} not found")

        with open(json_path, 'r') as f:
            data = json.load(f)

        experiment = PortfolioExperiment.from_dict(data)

        # Load portfolio history if available
        csv_path = self._safe_experiment_path(experiment_id, "_history.csv")
        if csv_path.exists():
            experiment.portfolio_history = pd.read_csv(csv_path)

        self.current_experiment = experiment
        return experiment

    def update_current(self, **kwargs):
        """Update current experiment with any fields."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment")

        for key, value in kwargs.items():
            if hasattr(self.current_experiment, key):
                setattr(self.current_experiment, key, value)

        self.current_experiment.updated_at = datetime.now().isoformat()
        self._save_experiment(self.current_experiment)

    def finish_experiment(self):
        """Finish current experiment."""
        if not self.current_experiment:
            raise RuntimeError("No active experiment")

        self._save_experiment(self.current_experiment)
        exp = self.current_experiment
        self.current_experiment = None
        return exp

    def query_experiments(
        self,
        tags: Optional[List[str]] = None,
        min_sharpe: Optional[float] = None,
        min_return: Optional[float] = None,
        algorithm: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Query experiments with filters.

        Args:
            tags: Filter by tags
            min_sharpe: Minimum Sharpe ratio
            min_return: Minimum total return
            algorithm: Filter by algorithm
            limit: Maximum results

        Returns:
            DataFrame of experiments
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if tags:
            placeholders = ','.join('?' * len(tags))
            query += f" AND experiment_id IN (SELECT experiment_id FROM tags WHERE tag IN ({placeholders}))"
            params.extend(tags)

        if min_sharpe is not None:
            query += " AND sharpe_ratio >= ?"
            params.append(min_sharpe)

        if min_return is not None:
            query += " AND total_return >= ?"
            params.append(min_return)

        if algorithm:
            query += " AND algorithm = ?"
            params.append(algorithm)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def get_best_experiments(self, metric: str = 'sharpe_ratio', top_n: int = 10) -> pd.DataFrame:
        """Get top experiments by metric."""
        if metric not in self.VALID_METRIC_COLUMNS:
            raise ValueError(
                f"Invalid metric: '{metric}'. "
                f"Valid metrics: {sorted(self.VALID_METRIC_COLUMNS)}"
            )
        conn = sqlite3.connect(self.db_path)

        query = f"""
            SELECT * FROM experiments
            WHERE {metric} IS NOT NULL
            ORDER BY {metric} DESC
            LIMIT ?
        """

        df = pd.read_sql_query(query, conn, params=(top_n,))
        conn.close()

        return df

    def delete_experiment(self, experiment_id: str):
        """Delete experiment."""
        json_path = self._safe_experiment_path(experiment_id, ".json")
        csv_path = self._safe_experiment_path(experiment_id, "_history.csv")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM experiments WHERE experiment_id = ?', (experiment_id,))
        cursor.execute('DELETE FROM tags WHERE experiment_id = ?', (experiment_id,))
        cursor.execute('DELETE FROM metrics_history WHERE experiment_id = ?', (experiment_id,))

        conn.commit()
        conn.close()

        if json_path.exists():
            json_path.unlink()
        if csv_path.exists():
            csv_path.unlink()
