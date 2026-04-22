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

## Project Structure

```
rlportfolio/
├── data/                   # Data fetching and feature engineering
│   ├── fetcher.py         # yfinance/Polygon data fetcher
│   └── features.py        # Technical indicators with registry pattern
├── environment/           # Custom Gym environment
│   ├── portfolio_env.py   # Multi-asset portfolio environment
│   ├── rewards.py         # Reward function implementations
│   └── transaction_costs.py # Pluggable cost/slippage models
├── training/              # Training infrastructure
│   ├── train.py           # Training script with config support
│   └── models/            # Saved models and checkpoints
├── evaluation/            # Backtesting and analysis
│   ├── metrics.py         # Performance metrics (Sharpe, Sortino, etc.)
│   ├── backtest.py        # Stateful backtester
│   ├── baselines.py       # Baseline strategy implementations
│   ├── visualization.py   # Plotting functions
│   └── visualize_network.py # NN architecture visualization
├── experiments/           # Experiment tracking (W&B, MLflow, SQLite)
├── configs/               # YAML configuration files
│   ├── default_config.yaml
│   ├── sac_config.yaml
│   └── tech_stocks_config.yaml
└── tests/                 # Unit tests
```

## Architecture

### Data Pipeline

1. **DataFetcher**: Downloads and caches historical OHLCV data from yfinance or Polygon
2. **FeatureEngineer**: Computes technical indicators using a registry pattern for extensibility
3. Features are normalized and prepared for the RL environment

### Environment

- **State Space**: Market features (prices, indicators) + current portfolio state (weights, cash)
- **Action Space**: Continuous portfolio weights (normalized via softmax)
- **Reward**: Configurable (Sharpe ratio, returns, risk-adjusted, etc.)
- **Transaction Costs**: Realistic trading costs applied during rebalancing

### Training

- Uses Stable-Baselines3 for RL algorithms
- Supports PPO (default), SAC, and A2C
- Configuration via YAML files
- Tensorboard logging and model checkpointing
- Evaluation callback for validation

### Evaluation

- **Metrics**: Total return, Sharpe ratio, Sortino ratio, max drawdown, volatility, win rate, etc.
- **Baselines**: Equal weight, buy-and-hold, momentum, minimum variance, inverse volatility
- **Visualization**: Performance comparison, drawdown, weights evolution, risk-return scatter

## Configuration

Configurations are stored in `configs/` as YAML files. Key parameters:

```yaml
data:
  tickers: [AAPL, MSFT, GOOGL, AMZN, NVDA]
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
from evaluation.baselines import BaselineStrategy

class MyStrategy(BaselineStrategy):
    def __init__(self):
        super().__init__('my_strategy')

    def get_action(self, env, step, **kwargs):
        # Your strategy logic
        return np.ones(env.n_assets)  # Equal weight example

# Register it
backtester = Backtester()
backtester.baseline_registry.register(MyStrategy())
```

## Data Sources

### yfinance (Default)
- Free, no API key required
- Daily data for stocks and ETFs
- Occasional data gaps

### Polygon.io
- Free tier: API key required (set `POLYGON_API_KEY` environment variable)
- Higher quality data
- Intraday support

## Results

Example performance metrics (will vary based on market conditions and training):

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Volatility |
|----------|--------------|--------------|--------------|------------|
| PPO Agent | 24.5% | 1.45 | -12.3% | 15.2% |
| SAC Agent | 22.1% | 1.38 | -11.8% | 14.8% |
| Equal Weight | 18.3% | 1.12 | -15.6% | 16.5% |
| Momentum | 20.8% | 1.25 | -18.2% | 17.8% |

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
- Market data from [yfinance](https://github.com/ranaroussi/yfinance)
