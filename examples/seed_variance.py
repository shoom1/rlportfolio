"""Overfitting stress test: seed variance + held-out test.

For each seed in SEEDS:
  - Train PPO on the full training window (from the given config).
  - Use the first half of the config's val window as an in-sample
    selection set for EvalCallback (saves best_model).
  - Report metrics on the second half of val (strictly held-out — never
    touched during training or model selection).

Aggregates mean / std / min / max Sharpe across seeds, plus same-window
buy_and_hold and equal_weight Sharpe for reference. Useful for deciding
whether a single-seed Sharpe edge is signal or noise.

Note: CFG_PATH and SEEDS are hardcoded for the div19 study. Adapt them
for other configs / seed sets.

Usage (from project root):
    conda run -n rlportfolio python examples/seed_variance.py
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from training.config import TrainingConfig
from training.train import PortfolioTrainer
from evaluation import Backtester


SEEDS = [42, 43, 44, 45, 46]
CFG_PATH = PROJECT_ROOT / "configs" / "opt_c_div19.yaml"


def split_by_date(env_data: pd.DataFrame, split_ratio: float = 0.5):
    """Split a MultiIndex (date, ticker) DataFrame by trading-day position."""
    dates = env_data.index.get_level_values('date').unique().sort_values()
    split_idx = int(len(dates) * split_ratio)
    split_date = dates[split_idx]
    early = env_data[env_data.index.get_level_values('date') < split_date]
    late = env_data[env_data.index.get_level_values('date') >= split_date]
    return early, late


def run_one_seed(seed: int, trainer: PortfolioTrainer,
                 train_data, val_early, val_late) -> dict:
    cfg = trainer.cfg
    train_env = trainer.create_environment(train_data)
    eval_env = trainer.create_environment(val_early)

    with tempfile.TemporaryDirectory() as tmpdir:
        best_dir = Path(tmpdir) / "best"
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            eval_freq=cfg.training.eval_freq,
            deterministic=True,
            n_eval_episodes=1,
            render=False,
            verbose=0,
        )

        agent = PPO(
            policy=cfg.agent.policy,
            env=train_env,
            verbose=0,
            seed=seed,
            **cfg.agent.extra,
        )
        agent.learn(
            total_timesteps=cfg.training.total_timesteps,
            callback=eval_callback,
        )

        best_path = best_dir / "best_model.zip"
        # Fall back to the trained agent if EvalCallback never produced a
        # checkpoint (small budgets can get to end of training before first eval).
        model = PPO.load(str(best_path)) if best_path.exists() else agent

    test_env = trainer.create_environment(val_late)
    bt = Backtester()
    bt.run_agent(model, test_env, name=f"PPO_seed{seed}")
    m = bt.compute_metrics(f"PPO_seed{seed}")
    return {
        'seed': seed,
        'sharpe': m['sharpe_ratio'],
        'total_return': m['total_return'],
        'max_dd': m['max_drawdown'],
        'volatility': m['volatility'],
    }


def reference_baselines(trainer: PortfolioTrainer, val_late) -> dict:
    """Run buy_and_hold and equal_weight on the held-out test window once."""
    bt = Backtester()
    for b in ("buy_and_hold", "equal_weight"):
        env = trainer.create_environment(val_late)
        bt.run_baseline(env, b)
    return {
        b: bt.compute_metrics(b)
        for b in ("buy_and_hold", "equal_weight")
    }


def main():
    cfg = TrainingConfig.from_yaml(CFG_PATH)
    trainer = PortfolioTrainer(cfg=cfg)
    train_data = trainer.prepare_data(train=True)
    val_data = trainer.prepare_data(train=False)
    val_early, val_late = split_by_date(val_data, split_ratio=0.5)

    early_dates = val_early.index.get_level_values('date')
    late_dates = val_late.index.get_level_values('date')
    print(f"Train: {len(train_data.index.get_level_values('date').unique())} dates")
    print(f"Val early (model selection): {early_dates.min().date()} -> "
          f"{early_dates.max().date()} ({len(early_dates.unique())} dates)")
    print(f"Val late  (held-out test):   {late_dates.min().date()} -> "
          f"{late_dates.max().date()} ({len(late_dates.unique())} dates)")

    results = []
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===", flush=True)
        r = run_one_seed(seed, trainer, train_data, val_early, val_late)
        print(r, flush=True)
        results.append(r)

    df = pd.DataFrame(results)
    base = reference_baselines(trainer, val_late)

    print("\n" + "=" * 70)
    print("PER-SEED RESULTS (held-out test window)")
    print("=" * 70)
    print(df.to_string(index=False,
                       formatters={
                           'sharpe': lambda x: f"{x:.3f}",
                           'total_return': lambda x: f"{x:.2%}",
                           'max_dd': lambda x: f"{x:.2%}",
                           'volatility': lambda x: f"{x:.2%}",
                       }))

    print("\n" + "=" * 70)
    print(f"AGGREGATE (n={len(SEEDS)} seeds)")
    print("=" * 70)
    print(f"PPO Sharpe:       mean={df['sharpe'].mean():.3f}  std={df['sharpe'].std():.3f}  "
          f"min={df['sharpe'].min():.3f}  max={df['sharpe'].max():.3f}")
    print(f"PPO Total return: mean={df['total_return'].mean():.2%}  std={df['total_return'].std():.2%}")
    print(f"PPO Max DD:       mean={df['max_dd'].mean():.2%}  std={df['max_dd'].std():.2%}")
    print(f"\nbuy_and_hold on same window:  Sharpe={base['buy_and_hold']['sharpe_ratio']:.3f}  "
          f"Return={base['buy_and_hold']['total_return']:.2%}")
    print(f"equal_weight on same window:  Sharpe={base['equal_weight']['sharpe_ratio']:.3f}  "
          f"Return={base['equal_weight']['total_return']:.2%}")


if __name__ == "__main__":
    main()
