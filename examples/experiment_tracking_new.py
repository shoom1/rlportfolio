"""
Example usage of the new experiment tracking architecture.

Demonstrates how to use different tracking strategies (W&B, MLflow, Mock)
with the callback pattern and context managers.
"""

import sys
sys.path.append('..')

from stable_baselines3 import PPO
from data.fetcher import DataFetcher
from data.features import FeatureEngineer
from environment.portfolio_env import PortfolioEnv
from environment.rewards import SharpeReward
from experiments import (
    WandbTracker,
    MLflowTracker,
    MockTracker,
    VerboseMockTracker,
    ExperimentTrackingCallback,
    MultiTrackerCallback
)


def create_environment():
    """Create a portfolio environment for training."""
    # Fetch data
    print("Fetching data...")
    fetcher = DataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = fetcher.get_latest_data(tickers, days=500)

    # Compute features
    print("Computing features...")
    engineer = FeatureEngineer()
    features_data = engineer.compute_features(data)
    env_data = engineer.prepare_for_environment(features_data)

    # Create environment
    feature_cols = engineer.create_observation_columns(normalize=True)
    env = PortfolioEnv(
        data=env_data,
        feature_columns=feature_cols,
        tickers=tickers,
        initial_balance=10000.0,
        transaction_cost=0.001,
        reward_function=SharpeReward(window=30),
        window_size=20
    )

    return env


def example_mock_tracker():
    """Example 1: Using MockTracker (no actual logging)."""
    print("\n" + "="*80)
    print("Example 1: Mock Tracker (Silent)")
    print("="*80)

    # Create environment and model
    env = create_environment()
    model = PPO("MlpPolicy", env, verbose=0)

    # Create mock tracker
    tracker = MockTracker(
        experiment_name="mock_experiment",
        config={"algorithm": "PPO", "learning_rate": 0.0003}
    )

    # Create callback
    callback = ExperimentTrackingCallback(tracker, log_freq=1000)

    # Train with context manager
    print("Training with MockTracker (nothing will be logged)...")
    with tracker.track():
        model.learn(total_timesteps=5000, callback=callback)

    print("✓ Training completed with MockTracker\n")


def example_verbose_mock_tracker():
    """Example 2: Using VerboseMockTracker (prints what would be logged)."""
    print("\n" + "="*80)
    print("Example 2: Verbose Mock Tracker (Prints Logs)")
    print("="*80)

    # Create environment and model
    env = create_environment()
    model = PPO("MlpPolicy", env, verbose=0)

    # Create verbose mock tracker
    tracker = VerboseMockTracker(
        experiment_name="verbose_mock_experiment",
        config={"algorithm": "PPO", "learning_rate": 0.0003}
    )

    # Create callback
    callback = ExperimentTrackingCallback(
        tracker,
        log_freq=1000,
        log_portfolio_weights=True
    )

    # Train with context manager
    print("Training with VerboseMockTracker (will print what it logs)...")
    with tracker.track():
        # Log hyperparameters
        tracker.log_hyperparameters({
            "total_timesteps": 5000,
            "batch_size": 64,
            "n_epochs": 10
        })

        model.learn(total_timesteps=5000, callback=callback)

    print("✓ Training completed with VerboseMockTracker\n")


def example_wandb_tracker():
    """Example 3: Using WandbTracker (requires wandb installation)."""
    print("\n" + "="*80)
    print("Example 3: W&B Tracker")
    print("="*80)

    try:
        # Create environment and model
        env = create_environment()
        model = PPO("MlpPolicy", env, verbose=0)

        # Create W&B tracker
        tracker = WandbTracker(
            experiment_name="wandb_ppo_experiment",
            config={
                "algorithm": "PPO",
                "learning_rate": 0.0003,
                "n_steps": 2048,
                "batch_size": 64
            }
        )

        # Create callback
        callback = ExperimentTrackingCallback(
            tracker,
            log_freq=1000,
            log_portfolio_weights=True,
            log_rollout_stats=True
        )

        # Train with context manager
        print("Training with WandbTracker...")
        with tracker.track(project="rl-portfolio-demo", mode="offline"):
            # Log hyperparameters
            tracker.log_hyperparameters({
                "total_timesteps": 10000,
                "tickers": ['AAPL', 'MSFT', 'GOOGL']
            })

            model.learn(total_timesteps=10000, callback=callback)

            # Save model and log as artifact
            model_path = "models/wandb_demo_model.zip"
            model.save(model_path)
            tracker.log_artifact(model_path, artifact_type="model")

        print("✓ Training completed with WandbTracker\n")

    except ImportError:
        print("⚠ wandb not installed. Skipping W&B example.")
        print("  Install with: pip install wandb\n")


def example_mlflow_tracker():
    """Example 4: Using MLflowTracker (requires mlflow installation)."""
    print("\n" + "="*80)
    print("Example 4: MLflow Tracker")
    print("="*80)

    try:
        # Create environment and model
        env = create_environment()
        model = PPO("MlpPolicy", env, verbose=0)

        # Create MLflow tracker
        tracker = MLflowTracker(
            experiment_name="mlflow_ppo_experiment",
            config={
                "algorithm": "PPO",
                "learning_rate": 0.0003,
                "n_steps": 2048
            }
        )

        # Create callback
        callback = ExperimentTrackingCallback(
            tracker,
            log_freq=1000,
            log_portfolio_weights=True
        )

        # Train with context manager
        print("Training with MLflowTracker...")
        with tracker.track(run_name="demo_run"):
            # Log hyperparameters
            tracker.log_hyperparameters({
                "total_timesteps": 10000,
                "tickers": ['AAPL', 'MSFT', 'GOOGL']
            })

            # Set tags
            tracker.set_tags({
                "model_type": "PPO",
                "environment": "portfolio"
            })

            model.learn(total_timesteps=10000, callback=callback)

            # Save model and log
            model_path = "models/mlflow_demo_model.zip"
            model.save(model_path)
            tracker.log_artifact(model_path, artifact_type="model")

        print("✓ Training completed with MLflowTracker\n")

    except ImportError:
        print("⚠ mlflow not installed. Skipping MLflow example.")
        print("  Install with: pip install mlflow\n")


def example_multi_tracker():
    """Example 5: Using multiple trackers simultaneously."""
    print("\n" + "="*80)
    print("Example 5: Multiple Trackers (W&B + MLflow)")
    print("="*80)

    # Create environment and model
    env = create_environment()
    model = PPO("MlpPolicy", env, verbose=0)

    # Use VerboseMockTracker to simulate both W&B and MLflow
    wandb_tracker = VerboseMockTracker(
        experiment_name="multi_wandb_experiment",
        config={"algorithm": "PPO"}
    )

    mlflow_tracker = VerboseMockTracker(
        experiment_name="multi_mlflow_experiment",
        config={"algorithm": "PPO"}
    )

    # Create multi-tracker callback
    callback = MultiTrackerCallback(
        trackers=[wandb_tracker, mlflow_tracker],
        log_freq=1000
    )

    # Train with both trackers
    print("Training with multiple trackers...")
    with wandb_tracker.track():
        with mlflow_tracker.track():
            model.learn(total_timesteps=5000, callback=callback)

    print("✓ Training completed with multiple trackers\n")


def example_custom_metrics():
    """Example 6: Logging custom portfolio metrics."""
    print("\n" + "="*80)
    print("Example 6: Custom Portfolio Metrics")
    print("="*80)

    # Create environment and model
    env = create_environment()
    model = PPO("MlpPolicy", env, verbose=0)

    # Create tracker
    tracker = VerboseMockTracker(
        experiment_name="custom_metrics_experiment"
    )

    # Create callback
    callback = ExperimentTrackingCallback(tracker, log_freq=1000)

    # Train
    with tracker.track():
        model.learn(total_timesteps=5000, callback=callback)

        # Log custom portfolio metrics
        tracker.log_portfolio_metrics(
            portfolio_value=10500.0,
            weights=[0.33, 0.33, 0.34],
            cash_ratio=0.01,
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            step=5000
        )

        # Log episode summary
        tracker.log_episode_summary(
            episode_reward=250.5,
            episode_length=100,
            final_portfolio_value=10500.0,
            sharpe_ratio=1.8,
            max_drawdown=0.05,
            step=5000
        )

    print("✓ Custom metrics logged\n")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Experiment Tracking Examples")
    print("="*80)

    # Run examples
    example_mock_tracker()
    example_verbose_mock_tracker()
    example_custom_metrics()
    example_multi_tracker()

    # These require optional dependencies
    # Uncomment to try them:
    # example_wandb_tracker()
    # example_mlflow_tracker()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
