# RL Portfolio Optimization

A reinforcement learning-based portfolio optimization system using Stable-Baselines3, designed to showcase modern RL techniques applied to quantitative finance.

## Overview

This project implements a multi-asset portfolio allocation agent using deep reinforcement learning. The agent learns to dynamically rebalance a portfolio to maximize risk-adjusted returns while accounting for transaction costs and market dynamics.

### Key Features

- **Modular Architecture**: Extensible design with registry patterns for features, rewards, and baseline strategies
- **Multiple RL Algorithms**: Support for PPO, SAC, and A2C from Stable-Baselines3
- **Rich Feature Engineering**: Technical indicators (RSI, MACD, Bollinger Bands, etc.) using pandas-ta
- **Flexible Reward Functions**: Sharpe ratio, Sortino ratio, risk-adjusted returns, and more
- **Comprehensive Backtesting**: Compare RL agents against traditional baselines (equal weight, momentum, min variance, etc.)
- **Walk-Forward Evaluation**: Expanding-window walk-forward harness (library + CLI) for honest out-of-sample assessment across regimes
- **Professional Evaluation**: Complete metrics suite and visualization tools

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd rlportfolio

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate rlportfolio
```

### Option 2: Using pip + venv

```bash
# Clone the repository
git clone <repository-url>
cd rlportfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Why conda?** Better dependency resolution for scientific computing packages (numpy, scipy, matplotlib) and easier management of platform-specific binaries.

### Development install

To work on the code and run the test suite, install the project in editable mode:

```bash
pip install -e .
```

This registers the `data`, `environment`, `evaluation`, `experiments`, and `training` packages on `sys.path` so imports resolve without needing a working directory hack. Run the tests with:

```bash
pytest tests/
```

## Quick Start

### 1. Train an Agent

```bash
# Train with default configuration (PPO on 5 tech stocks)
python training/train.py

# Train with custom configuration
python training/train.py --config configs/sac_config.yaml

# Resume from checkpoint
python training/train.py --resume training/models/portfolio_agent_50000_steps.zip
```

### 2. Evaluate a Trained Model

```bash
python training/train.py --eval training/models/best/best_model.zip
```

### 3. Run Backtests and Compare Strategies

```python
from data.fetcher import DataFetcher
from data.features import FeatureEngineer
from environment import PortfolioEnv
from evaluation import Backtester, plot_strategy_comparison
from stable_baselines3 import PPO

# Prepare data
fetcher = DataFetcher()
engineer = FeatureEngineer()
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

data = fetcher.get_latest_data(tickers, days=500)
features = engineer.compute_features(data)
env_data = engineer.prepare_for_environment(features)

# Create environment
feature_cols = engineer.create_observation_columns()
env = PortfolioEnv(
    data=env_data,
    feature_columns=feature_cols,
    tickers=tickers
)

# Load trained agent
agent = PPO.load('training/models/best/best_model.zip')

# Run backtests
backtester = Backtester()
backtester.run_agent(agent, env, name='PPO')
backtester.run_baseline(env, 'equal_weight')
backtester.run_baseline(env, 'momentum_20')
backtester.run_baseline(env, 'min_variance_60')

# Compare results
backtester.print_comparison()

# Visualize
histories = backtester.get_histories()
plot_strategy_comparison(histories, save_path='results/comparison.png')
```

### 4. Walk-Forward Out-of-Sample Evaluation

For a rigorous OOS assessment across market regimes — trains a fresh agent on every fold's expanding window and backtests on the next disjoint slice:

```bash
# Quarterly walk-forward from 2005 to today, ~70 folds
conda run -n rlportfolio python -m evaluation.walk_forward \
    --config configs/opt_c_div19.yaml \
    --t-min-days 756 --stride-days 63 \
    --seeds 42 43 44 \
    --output results/walk_forward.csv
```

Produces a per-fold/per-seed CSV (agent + baselines on each disjoint test window), records failed runs for investigation, and prints aggregate run-level and fold-mean Sharpe / hit-rate summaries. The last 6 months of each fold's train window are reserved for model selection via `EvalCallback`, so the test window is never seen during training. Programmatic API in `evaluation.walk_forward.WalkForwardEvaluator`; see `examples/walk_forward.py` for a minimal driver.

## Project Structure

```
rlportfolio/
├── data/                     # Data fetching and feature engineering
│   ├── fetcher.py            # Thin adapter around finbase.DataClient
│   └── features.py           # Technical indicators with registry pattern
├── environment/              # Custom Gymnasium environment
│   ├── portfolio_env.py      # Multi-asset portfolio env (precomputed feature cube)
│   ├── rewards.py            # Reward function implementations
│   ├── transaction_costs.py  # Pluggable cost/slippage models
│   └── constants.py          # Shared numeric constants
├── training/                 # Training infrastructure
│   ├── train.py              # Training script + PortfolioTrainer
│   ├── config.py             # Typed TrainingConfig dataclasses
│   └── models/               # Saved models and checkpoints
├── evaluation/               # Backtesting and analysis
│   ├── metrics.py            # Performance metrics (Sharpe, Sortino, etc.)
│   ├── backtest.py           # Stateful Backtester facade
│   ├── backtest_strategies.py # Sequential / walk-forward / Monte Carlo execution
│   ├── baselines.py          # Baseline strategy implementations
│   ├── walk_forward.py       # Walk-forward training + OOS harness (library + CLI)
│   ├── visualization.py      # Plotting functions
│   └── visualize_network.py  # NN architecture visualization
├── experiments/              # Experiment tracking (W&B, MLflow, SQLite)
├── configs/                  # YAML configuration files (see directory for full list)
├── examples/                 # Thin demo scripts (walk_forward, seed_variance, ...)
└── tests/                    # Unit tests (320 passing)
```

## Architecture

### Data Pipeline

1. **DataFetcher**: Thin adapter over `finbase.DataClient`, which reads from a
   shared SQLite database at `~/.finbase/timeseries.db`. Data is populated and
   refreshed by the `finbase` project — this repo only reads it.
2. **FeatureEngineer**: Computes technical indicators using a registry pattern for extensibility
3. Features are normalized and prepared for the RL environment

### Environment

- **State Space**: Market features (prices, indicators) + current portfolio state (weights, cash)
- **Action Space**: Continuous portfolio weights (normalized via softmax)
- **Reward**: Configurable (Sharpe ratio, returns, risk-adjusted, etc.)
- **Transaction Costs**: Proportional costs are applied during rebalancing.
  Slippage defaults to zero for backward compatibility; custom cost models can
  use fixed, volume-based, or spread-based slippage, with `volume`,
  `bid_ask_spread`, or `spread` columns passed into trade records when present.

### Training

- Uses Stable-Baselines3 for RL algorithms
- Supports PPO (default), SAC, and A2C
- Configuration via YAML files
- Tensorboard logging and model checkpointing
- Evaluation callback for validation
- **Disjoint train/val**: `PortfolioTrainer.prepare_data` fetches a single combined window of `train_days + val_days` and slices on date, so val is strictly after train (no in-sample leakage into the eval callback).

### Evaluation

- **Metrics**: Total return, Sharpe ratio, Sortino ratio, max drawdown, volatility, win rate, etc.
- **Baselines**: Equal weight, buy-and-hold, momentum, minimum variance, inverse volatility
- **Walk-Forward**: Expanding-window protocol that retrains per fold; last 6 months of each fold's train reserved for model selection; disjoint quarterly test windows. Library (`evaluation.walk_forward.WalkForwardEvaluator`) and CLI (`python -m evaluation.walk_forward`).
- **Visualization**: Performance comparison, drawdown, weights evolution, risk-return scatter

## Configuration

Configurations are stored in `configs/` as YAML files. Key parameters:

```yaml
data:
  tickers: [AAPL, MSFT, GOOGL, AMZN, NVDA]
  universe:
    mode: static_current
    survivorship_bias: known
  train_days: 730

environment:
  initial_balance: 10000.0
  transaction_cost: 0.001
  reward_function: sharpe

agent:
  algorithm: PPO
  learning_rate: 0.0003
  policy_kwargs:
    net_arch: [256, 256, 128]

training:
  total_timesteps: 100000
```

`data.universe.mode: static_current` is the only supported universe policy
today. It reuses the configured ticker list across every period and marks
outputs with `survivorship_bias: known`; it does not reconstruct historical
index membership. `point_in_time_index` is reserved for future support and
fails validation instead of silently behaving like a static universe.

## Extending the Framework

### Add Custom Features

```python
from data.features import Feature

class MyCustomFeature(Feature):
    def __init__(self):
        super().__init__('my_feature')

    def compute(self, df):
        df['my_indicator'] = df['Close'].rolling(10).mean()
        return df

    def get_column_names(self):
        return ['my_indicator']

# Use it
engineer = FeatureEngineer(custom_features=[MyCustomFeature()])
```

### Add Custom Reward Functions

```python
from environment.rewards import RewardFunction

class MyReward(RewardFunction):
    def compute(self, portfolio_return, portfolio_value, previous_value, **kwargs):
        # Your custom logic
        return portfolio_return * 2  # Example
```

### Add Custom Baseline Strategies

```python
import numpy as np

from environment.constants import CASH_SOFTMAX_BIAS
from evaluation.backtest import Backtester
from evaluation.baselines import BaselineStrategy

class MyStrategy(BaselineStrategy):
    def __init__(self):
        super().__init__('my_strategy')

    def get_action(self, env, step, **kwargs):
        action = np.ones(env.n_assets + 1)
        action[-1] = CASH_SOFTMAX_BIAS
        return action

# Register it
backtester = Backtester()
backtester.baseline_registry.register(MyStrategy())
```

## Data Sources

Market data is sourced through **`finbase.DataClient`**, a shared SQLite-backed
client used across the FinAI projects. The local database lives at
`~/.finbase/timeseries.db`. Populating and refreshing that database is the
responsibility of the `finbase` package; this repo is a read-only consumer.

See `data/fetcher.py` for the thin adapter layer.

## Results

Results are sensitive to universe, time period, and random seed. Single-window comparisons can overstate the RL edge — an honest assessment needs walk-forward across regimes. Current configs use `static_current` universes, so index-style claims should be treated as survivorship-biased unless you provide a true point-in-time membership source outside this package.

Run the walk-forward harness on your universe of choice to generate per-fold metrics:

```bash
conda run -n rlportfolio python -m evaluation.walk_forward \
    --config configs/opt_c_div19.yaml \
    --seeds 42 43 44
```

Outputs a per-fold/per-seed CSV and prints aggregate run-level and fold-mean Sharpe / hit-rate summaries. Each row includes `universe_mode`, `survivorship_bias`, `n_assets`, and `tickers_hash` so downstream analysis keeps the universe assumption attached to the result. Quarterly protocol with 3-year minimum train, 1-quarter stride and disjoint test windows, 6-month in-sample selection slice. See `evaluation/walk_forward.py` for all knobs.

## Requirements

- Python 3.8 - 3.12 (recommended: 3.11 or 3.12)
- stable-baselines3 >= 2.7.0
- gymnasium >= 1.0.0
- See `requirements.txt` or `environment.yml` for full dependencies

## License

MIT License

## Acknowledgments

- Built with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Technical indicators from [pandas-ta](https://github.com/twopirllc/pandas-ta)
- Market data via the internal `finbase` SQLite client
