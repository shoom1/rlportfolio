"""
Training script for portfolio optimization using Stable-Baselines3.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.append('..')

from data.fetcher import DataFetcher
from data.features import FeatureEngineer
from environment.portfolio_env import PortfolioEnv
from environment.rewards import RewardFunctionFactory
from experiments import (
    ExperimentTracker,
    MockTracker,
    WandbTracker,
    MLflowTracker,
    ExperimentTrackingCallback
)
from evaluation.metrics import PortfolioMetrics
from evaluation.baselines import BaselineStrategyRegistry
from evaluation.backtest import run_baseline_backtest

from typing import Optional


class PortfolioTrainer:
    """Trainer for portfolio optimization agents."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        tracker: Optional[ExperimentTracker] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize trainer with configuration.

        Args:
            config_path: Path to YAML configuration file
            tracker: ExperimentTracker instance (W&B, MLflow, Mock, etc.)
                    If None, uses MockTracker (no logging)
            experiment_name: Custom experiment name
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()

        # Experiment tracking
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tracker = tracker or MockTracker(self.experiment_name, config=self.config)

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'data': {
                'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
                'train_days': 730,  # 2 years
                'val_days': 180,    # 6 months
                'source': 'yfinance'
            },
            'environment': {
                'initial_balance': 10000.0,
                'transaction_cost': 0.001,
                'window_size': 20,
                'reward_function': 'sharpe',
                'reward_params': {
                    'window': 30,
                    'risk_free_rate': 0.02
                }
            },
            'agent': {
                'algorithm': 'PPO',
                'policy': 'MlpPolicy',
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'policy_kwargs': {
                    'net_arch': [256, 256, 128]
                }
            },
            'training': {
                'total_timesteps': 100000,
                'eval_freq': 5000,
                'save_freq': 10000,
                'log_dir': 'logs',
                'model_dir': 'agents/models'
            }
        }

    def prepare_data(self, train: bool = True) -> pd.DataFrame:
        """
        Fetch and prepare data for training or validation.

        Args:
            train: Whether to prepare training data (True) or validation data (False)

        Returns:
            Processed DataFrame ready for environment
        """
        tickers = self.config['data']['tickers']
        days = self.config['data']['train_days'] if train else self.config['data']['val_days']
        source = self.config['data']['source']

        print(f"Fetching {'training' if train else 'validation'} data...")
        data = self.data_fetcher.get_latest_data(tickers, days=days, source=source)

        print("Computing features...")
        features_data = self.feature_engineer.compute_features(data)

        print("Preparing for environment...")
        env_data = self.feature_engineer.prepare_for_environment(features_data)

        return env_data

    def create_environment(self, data: pd.DataFrame, monitor_path: Optional[str] = None) -> PortfolioEnv:
        """
        Create portfolio environment.

        Args:
            data: Prepared data
            monitor_path: Path for Monitor wrapper

        Returns:
            Portfolio environment
        """
        tickers = self.config['data']['tickers']
        feature_cols = self.feature_engineer.create_observation_columns(normalize=True)

        # Create reward function
        reward_fn_name = self.config['environment']['reward_function']
        reward_params = self.config['environment'].get('reward_params', {})
        reward_function = RewardFunctionFactory.create(reward_fn_name, **reward_params)

        # Create environment
        env = PortfolioEnv(
            data=data,
            feature_columns=feature_cols,
            tickers=tickers,
            initial_balance=self.config['environment']['initial_balance'],
            transaction_cost=self.config['environment']['transaction_cost'],
            reward_function=reward_function,
            window_size=self.config['environment']['window_size']
        )

        # Wrap with Monitor
        if monitor_path:
            env = Monitor(env, monitor_path)

        return env

    def train(self, resume_path: Optional[str] = None):
        """
        Train the agent.

        Args:
            resume_path: Path to checkpoint to resume from
        """
        # Prepare data
        train_data = self.prepare_data(train=True)
        val_data = self.prepare_data(train=False)

        # Create directories
        log_dir = Path(self.config['training']['log_dir'])
        model_dir = Path(self.config['training']['model_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create environments
        print("Creating environments...")
        train_env = self.create_environment(train_data, monitor_path=str(log_dir / "train"))
        eval_env = self.create_environment(val_data, monitor_path=str(log_dir / "eval"))

        # Create agent
        algorithm = self.config['agent']['algorithm']
        print(f"Creating {algorithm} agent...")

        agent_class = {
            'PPO': PPO,
            'SAC': SAC,
            'A2C': A2C
        }[algorithm]

        # Prepare agent kwargs
        agent_kwargs = {
            'policy': self.config['agent']['policy'],
            'env': train_env,
            'verbose': 1,
        }

        # Only add tensorboard if it's installed
        try:
            import tensorboard
            agent_kwargs['tensorboard_log'] = str(log_dir / 'tensorboard')
        except ImportError:
            print("Note: tensorboard not installed. Tensorboard logging disabled.")

        # Add algorithm-specific parameters
        for key in ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
                    'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm']:
            if key in self.config['agent']:
                agent_kwargs[key] = self.config['agent'][key]

        if 'policy_kwargs' in self.config['agent']:
            agent_kwargs['policy_kwargs'] = self.config['agent']['policy_kwargs']

        # Create or load agent
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            agent = agent_class.load(resume_path, env=train_env)
        else:
            agent = agent_class(**agent_kwargs)

        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['training']['save_freq'],
            save_path=str(model_dir),
            name_prefix='portfolio_agent'
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / 'best'),
            log_path=str(log_dir / 'eval'),
            eval_freq=self.config['training']['eval_freq'],
            deterministic=True,
            render=False
        )

        # Experiment tracking callback
        tracking_callback = ExperimentTrackingCallback(
            tracker=self.tracker,
            log_freq=1000,
            log_portfolio_weights=True,
            log_rollout_stats=True,
            verbose=1
        )

        callbacks = [checkpoint_callback, eval_callback, tracking_callback]

        # Train with context manager for tracking
        print(f"\nStarting training for {self.config['training']['total_timesteps']} timesteps...")

        # Determine tracker init kwargs based on tracker type
        tracker_init_kwargs = {}
        if isinstance(self.tracker, WandbTracker):
            tracker_init_kwargs = {
                'project': 'rl-portfolio',
                'tags': [algorithm, self.experiment_name]
            }
        elif isinstance(self.tracker, MLflowTracker):
            tracker_init_kwargs = {
                'run_name': self.experiment_name,
                'tags': {
                    'algorithm': algorithm,
                    'tickers': ','.join(self.config['data']['tickers'])
                }
            }

        with self.tracker.track(**tracker_init_kwargs):
            # Log configuration as hyperparameters
            self.tracker.log_hyperparameters(self.config)

            # Train
            agent.learn(
                total_timesteps=self.config['training']['total_timesteps'],
                callback=callbacks,
                progress_bar=True
            )

            # Save final model
            final_path = model_dir / 'final_model'
            agent.save(final_path)
            print(f"\nTraining completed! Final model saved to: {final_path}")

            # Log final model as artifact
            self.tracker.log_artifact(str(final_path) + '.zip', artifact_type='model')

        return agent

    def evaluate(self, model_path: str, render: bool = False):
        """
        Evaluate a trained model.

        Args:
            model_path: Path to trained model
            render: Whether to render environment
        """
        # Prepare validation data
        val_data = self.prepare_data(train=False)

        # Create environment
        eval_env = self.create_environment(val_data)

        # Load model
        algorithm = self.config['agent']['algorithm']
        agent_class = {
            'PPO': PPO,
            'SAC': SAC,
            'A2C': A2C
        }[algorithm]

        print(f"Loading model from: {model_path}")
        agent = agent_class.load(model_path)

        # Run evaluation
        print("Running evaluation...")
        obs, info = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            if render:
                eval_env.render()

        # Get portfolio history
        history = eval_env.get_portfolio_history()

        print(f"\nEvaluation Results:")
        print(f"Total Steps: {steps}")
        print(f"Total Reward: {total_reward:.4f}")
        print(f"Final Portfolio Value: ${history['value'].iloc[-1]:.2f}")
        print(f"Total Return: {(history['value'].iloc[-1] / history['value'].iloc[0] - 1) * 100:.2f}%")

        return history


def main():
    parser = argparse.ArgumentParser(description='Train portfolio optimization agent')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--eval', type=str, help='Path to model to evaluate')
    parser.add_argument('--render', action='store_true', help='Render during evaluation')
    parser.add_argument('--tracker', type=str, default='mock',
                       choices=['mock', 'wandb', 'mlflow', 'verbose'],
                       help='Experiment tracker to use (default: mock)')
    parser.add_argument('--experiment-name', type=str, help='Custom experiment name')

    args = parser.parse_args()

    # Create tracker based on argument
    experiment_name = args.experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.tracker == 'wandb':
        try:
            tracker = WandbTracker(experiment_name=experiment_name)
            print("Using Weights & Biases tracker")
        except ImportError:
            print("Warning: wandb not installed. Using MockTracker instead.")
            tracker = MockTracker(experiment_name=experiment_name)
    elif args.tracker == 'mlflow':
        try:
            tracker = MLflowTracker(experiment_name=experiment_name)
            print("Using MLflow tracker")
        except ImportError:
            print("Warning: mlflow not installed. Using MockTracker instead.")
            tracker = MockTracker(experiment_name=experiment_name)
    elif args.tracker == 'verbose':
        from experiments import VerboseMockTracker
        tracker = VerboseMockTracker(experiment_name=experiment_name)
        print("Using VerboseMockTracker (prints logs to console)")
    else:  # mock
        tracker = MockTracker(experiment_name=experiment_name)
        print("Using MockTracker (no logging)")

    # Create trainer with selected tracker
    trainer = PortfolioTrainer(
        config_path=args.config,
        tracker=tracker,
        experiment_name=experiment_name
    )

    if args.eval:
        trainer.evaluate(args.eval, render=args.render)
    else:
        trainer.train(resume_path=args.resume)


if __name__ == "__main__":
    main()
