"""
Experiment tracking and management for portfolio optimization.

Uses the Strategy pattern for pluggable experiment tracking backends.
"""

from .base import (
    ExperimentTracker,
    MockTracker,
    VerboseMockTracker
)
from .wandb_tracker import WandbTracker
from .mlflow_tracker import MLflowTracker
from .tracking_callback import (
    ExperimentTrackingCallback,
    MultiTrackerCallback
)

__all__ = [
    'ExperimentTracker',
    'MockTracker',
    'VerboseMockTracker',
    'WandbTracker',
    'MLflowTracker',
    'ExperimentTrackingCallback',
    'MultiTrackerCallback',
]
