"""
Unit tests for experiments/tracker.py - PortfolioExperimentTracker.
"""

import pytest
from experiments.portfolio_tracker import PortfolioExperimentTracker


class TestGetBestExperiments:
    """Tests for get_best_experiments metric validation."""

    def _create_tracker(self, tmp_path):
        """Create a tracker with a temporary storage directory."""
        return PortfolioExperimentTracker(storage_dir=str(tmp_path))

    def test_valid_metric(self, tmp_path):
        """Test that valid metrics are accepted."""
        tracker = self._create_tracker(tmp_path)
        exp = tracker.create_experiment("test_exp")
        exp.portfolio_metrics['sharpe_ratio'] = 1.5
        tracker.finish_experiment()

        result = tracker.get_best_experiments(metric='sharpe_ratio', top_n=10)
        assert len(result) == 1

    def test_all_valid_metrics_accepted(self, tmp_path):
        """Test that all whitelisted metrics are accepted without error."""
        tracker = self._create_tracker(tmp_path)
        for metric in PortfolioExperimentTracker.VALID_METRIC_COLUMNS:
            result = tracker.get_best_experiments(metric=metric, top_n=10)
            assert result is not None

    def test_invalid_metric_raises(self, tmp_path):
        """Test that invalid metric names raise ValueError."""
        tracker = self._create_tracker(tmp_path)
        with pytest.raises(ValueError, match="Invalid metric"):
            tracker.get_best_experiments(metric="1; DROP TABLE experiments")

    def test_invalid_metric_arbitrary_string_raises(self, tmp_path):
        """Test that arbitrary strings are rejected."""
        tracker = self._create_tracker(tmp_path)
        with pytest.raises(ValueError, match="Invalid metric"):
            tracker.get_best_experiments(metric="nonexistent_column")

    def test_respects_top_n(self, tmp_path):
        """Test that top_n limits the number of results."""
        from experiments.portfolio_tracker import PortfolioExperiment

        tracker = self._create_tracker(tmp_path)

        for i in range(3):
            exp = PortfolioExperiment(name=f"exp_{i}", experiment_id=f"test_exp_{i}")
            exp.set_portfolio_metrics({'sharpe_ratio': float(i + 1)})
            tracker._save_experiment(exp)

        result = tracker.get_best_experiments(metric='sharpe_ratio', top_n=2)
        assert len(result) == 2
