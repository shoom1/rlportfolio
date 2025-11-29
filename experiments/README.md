# Experiment Tracking for RL Portfolio Optimization

Comprehensive experiment tracking system combining **Weights & Biases** (for general RL metrics) and a **custom portfolio tracker** (for domain-specific analysis).

## Features

### 🎯 Dual Tracking System

1. **Weights & Biases Integration**
   - Real-time training metrics
   - Hyperparameter sweeps
   - Model versioning
   - Beautiful dashboards
   - Team collaboration

2. **Custom Portfolio Tracker**
   - Portfolio-specific metrics (Sharpe, Sortino, drawdown)
   - SQLite database for fast queries
   - Baseline strategy comparisons
   - Portfolio weight evolution
   - Offline-first with full control

### 📊 Key Capabilities

- **Automatic Logging**: Tracks all metrics during training
- **Baseline Comparison**: Compare against buy-and-hold, equal-weight, etc.
- **Experiment Comparison**: Rank and compare multiple experiments
- **Rich CLI**: Terminal UI for experiment management
- **Export Tools**: CSV, plots, comprehensive reports
- **Hyperparameter Sweeps**: Automated with W&B

## Quick Start

### 1. Install Dependencies

```bash
conda run -n rlportfolio pip install -r requirements.txt
```

### 2. Basic Usage

```python
from agents.train_with_tracking import create_trainer_with_tracking

# Create trainer with tracking
trainer = create_trainer_with_tracking(
    config_path='configs/ppo_config.yaml',
    use_wandb=True,  # Enable W&B
    wandb_project='my-portfolio-project',
    experiment_name='ppo_5stocks_experiment'
)

# Train (tracking happens automatically)
agent = trainer.train(tags=['baseline', 'ppo'])
```

### 3. View Results

```bash
# List all experiments
python -m experiments.cli list

# Show experiment details
python -m experiments.cli show <experiment_id>

# Compare experiments
python -m experiments.cli compare exp1 exp2 exp3 --export ./comparison

# Rank experiments
python -m experiments.cli rank --top-n 10

# Show best by Sharpe ratio
python -m experiments.cli best --metric sharpe_ratio --top-n 5
```

## Architecture

### Components

```
experiments/
├── tracker.py              # Custom portfolio experiment tracker (SQLite)
├── experiment_tracker.py   # Base tracker interface (Strategy pattern)
├── wandb_tracker.py        # W&B integration using Strategy pattern
├── mlflow_tracker.py       # MLflow integration using Strategy pattern
├── tracking_callback.py    # Multi-tracker callbacks for SB3
├── comparison.py           # Tools for comparing experiments
├── cli.py                  # Command-line interface
└── README.md              # This file
```

### Data Storage

```
experiments/
└── portfolio/
    ├── experiments.db           # SQLite database (fast queries)
    ├── exp_20241127_143022.json # Detailed experiment data
    ├── exp_20241127_143022_history.csv  # Portfolio history
    └── ...
```

## Tracked Metrics

### General RL Metrics (W&B)
- Episode reward, length
- Policy loss, value loss
- Entropy, explained variance
- Learning rate, clipfrac
- FPS, iterations

### Portfolio-Specific Metrics (Custom Tracker)
- **Returns**: Total return, annualized return
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Risk**: Volatility, max drawdown, downside deviation
- **Trading**: Win rate, profit factor, transaction costs
- **Weights**: Portfolio allocation over time

## Advanced Usage

### 1. Hyperparameter Sweeps with W&B

```python
import wandb
from experiments import create_wandb_sweep_config

# Define sweep configuration
sweep_config = create_wandb_sweep_config(
    base_config={},
    parameters_to_sweep={
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': 1e-5,
            'max': 1e-2
        },
        'n_steps': {
            'values': [1024, 2048, 4096]
        },
        'ent_coef': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.1
        }
    }
)

# Create sweep
sweep_id = wandb.sweep(sweep_config, project='rl-portfolio')

# Run sweep
def train_sweep():
    trainer = create_trainer_with_tracking(
        use_wandb=True,
        wandb_project='rl-portfolio'
    )
    trainer.train()

wandb.agent(sweep_id, function=train_sweep, count=20)
```

### 2. Custom Experiment Queries

```python
from experiments import PortfolioExperimentTracker

tracker = PortfolioExperimentTracker()

# Query experiments
results = tracker.query_experiments(
    tags=['production'],
    min_sharpe=1.0,
    min_return=0.1,
    algorithm='PPO'
)

# Get best experiments
best = tracker.get_best_experiments(
    metric='sharpe_ratio',
    top_n=10
)

print(best[['name', 'sharpe_ratio', 'total_return']])
```

### 3. Programmatic Comparison

```python
from experiments import ExperimentComparison, PortfolioExperimentTracker

tracker = PortfolioExperimentTracker()
comparison = ExperimentComparison(tracker)

# Compare experiments
df = comparison.compare_experiments(
    experiment_ids=['exp1', 'exp2', 'exp3'],
    metrics=['sharpe_ratio', 'total_return', 'max_drawdown']
)

# Generate visualizations
comparison.plot_comparison(
    experiment_ids=['exp1', 'exp2', 'exp3'],
    metric='sharpe_ratio'
)

comparison.plot_portfolio_values(
    experiment_ids=['exp1', 'exp2', 'exp3']
)

# Rank experiments
ranked = comparison.rank_experiments(
    experiment_ids=['exp1', 'exp2', 'exp3'],
    weights={
        'sharpe_ratio': 0.4,
        'total_return': 0.3,
        'calmar_ratio': 0.3
    }
)

# Export comparison
comparison.export_comparison(
    experiment_ids=['exp1', 'exp2', 'exp3'],
    output_dir='comparison_results'
)
```

### 4. Manual Tracking (Advanced)

```python
from experiments import PortfolioExperimentTracker

tracker = PortfolioExperimentTracker()

# Create experiment
exp = tracker.create_experiment(
    name='my_custom_experiment',
    description='Testing new reward function',
    tags=['custom', 'test'],
    config={'my_param': 42}
)

# Update during training
exp.model_info.update({
    'algorithm': 'PPO',
    'policy': 'MlpPolicy'
})

# Set metrics after evaluation
exp.set_portfolio_metrics({
    'sharpe_ratio': 1.5,
    'total_return': 0.25,
    'max_drawdown': -0.12
})

# Save
tracker.finish_experiment()
```

## CLI Reference

### List Experiments
```bash
python -m experiments.cli list [--tags TAG1 TAG2] [--algorithm PPO] \
    [--min-sharpe 1.0] [--min-return 0.1] [--limit 50]
```

### Show Experiment Details
```bash
python -m experiments.cli show <experiment_id>
```

### Compare Experiments
```bash
python -m experiments.cli compare exp1 exp2 exp3 \
    [--output report.txt] [--export ./comparison_dir]
```

### Rank Experiments
```bash
python -m experiments.cli rank [exp1 exp2 exp3] [--top-n 10]
```

### Show Best Experiments
```bash
python -m experiments.cli best [--metric sharpe_ratio] [--top-n 10]
```

### Delete Experiment
```bash
python -m experiments.cli delete <experiment_id> [--force]
```

## W&B Dashboard

When W&B is enabled, you can view:

1. **Training Metrics**: Real-time loss, reward, etc.
2. **Portfolio Charts**: Value evolution, weight distribution
3. **Comparison Tables**: Agent vs baselines
4. **Hyperparameter Impact**: Parallel coordinates, scatter plots
5. **Model Artifacts**: Saved checkpoints

Access at: https://wandb.ai/your-username/rl-portfolio

## Example Workflow

### Complete Training & Analysis Workflow

```bash
# 1. Train with experiment tracking
python agents/train_with_tracking.py --config configs/ppo_config.yaml

# 2. List recent experiments
python -m experiments.cli list --limit 10

# 3. Show details of best experiment
BEST_EXP=$(python -m experiments.cli best --top-n 1 | grep -oP "exp_\d+_\d+")
python -m experiments.cli show $BEST_EXP

# 4. Compare top 3 experiments
python -m experiments.cli rank --top-n 3
python -m experiments.cli compare exp1 exp2 exp3 --export ./comparison

# 5. View W&B dashboard
# Open https://wandb.ai in browser
```

## Configuration

### W&B Settings

```python
# In your training script
trainer = create_trainer_with_tracking(
    use_wandb=True,
    wandb_project='my-project',
    experiment_name='ppo_experiment_1'
)

# Or disable W&B
trainer = create_trainer_with_tracking(
    use_wandb=False  # Only use custom tracker
)
```

### Storage Location

```python
# Custom storage directory
from experiments import PortfolioExperimentTracker

tracker = PortfolioExperimentTracker(
    storage_dir='my_custom_experiments'
)
```

## Best Practices

1. **Tag Your Experiments**: Use tags for easy filtering
   ```python
   trainer.train(tags=['baseline', 'production', 'v1.0'])
   ```

2. **Meaningful Names**: Use descriptive experiment names
   ```python
   experiment_name='ppo_5stocks_sharpe_reward_v2'
   ```

3. **Regular Cleanup**: Delete failed/obsolete experiments
   ```bash
   python -m experiments.cli list | grep failed
   python -m experiments.cli delete <exp_id>
   ```

4. **Export Important Results**: Save comparisons for reports
   ```bash
   python -m experiments.cli compare exp1 exp2 --export ./paper_results
   ```

5. **Use Sweeps for Tuning**: Let W&B find best hyperparameters
   ```python
   # See hyperparameter sweep example above
   ```

## Troubleshooting

### W&B Login Issues
```bash
wandb login
# Or set environment variable
export WANDB_API_KEY=your_key_here
```

### Offline Mode
```python
# Use W&B in offline mode
import os
os.environ['WANDB_MODE'] = 'offline'
```

### Database Locked
```python
# If SQLite is locked, increase timeout
import sqlite3
conn = sqlite3.connect('experiments.db', timeout=30.0)
```

## Integration with Existing Code

The tracking system is designed to be **non-invasive**:

- ✅ Works with existing `train.py` (use `train_with_tracking.py`)
- ✅ No changes required to environment or agents
- ✅ Can be disabled completely (`use_wandb=False`)
- ✅ Backward compatible with existing logs

## Future Enhancements

- [ ] Web dashboard for experiment browsing
- [ ] Automated A/B testing framework
- [ ] Integration with MLflow
- [ ] Real-time Slack/email notifications
- [ ] Experiment reproduction from ID
- [ ] Automated report generation

## Support

For issues or questions:
1. Check this README
2. View example usage in `examples/`
3. Open an issue on GitHub
