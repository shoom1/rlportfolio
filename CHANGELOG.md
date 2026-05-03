# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-03

### Added

- Walk-forward evaluation harness (`evaluation/walk_forward.py`,
  library + CLI). Expanding-window protocol, disjoint test slices,
  parallel `ProcessPoolExecutor` workers, multi-seed support.
- Pluggable transaction-cost and slippage models
  (`environment/transaction_costs.py`): proportional / tiered /
  nonlinear costs, fixed / volume / spread slippage, combined model,
  registry.
- Experiment tracking package (`experiments/`): W&B, MLflow, SQLite
  trackers (Strategy pattern), comparison CLI, multi-tracker callback.
- Typed `TrainingConfig` dataclasses (`training/config.py`) with
  YAML / dict / programmatic precedence and `__post_init__`
  validation.
- Disjoint train/val: `PortfolioTrainer.prepare_data` slices a
  single fetch by date.
- Static-universe metadata: `data.universe.mode`,
  `survivorship_bias`, `tickers_hash` recorded per run.
- `BaselineStrategyRegistry.create(kind, **kwargs)` for parametric
  baseline construction.
- `scripts/results_table.py` — render a walk-forward CSV as a
  markdown table with optional in-place README patching.
- `scripts/add_market_benchmark.py` — append `^GSPC` (or any
  ticker) Sharpe / return columns to a walk-forward CSV.
- `scripts/plot_walk_forward.py` — render four PNGs (equity curves,
  Sharpe boxplot, rolling Sharpe, rolling hit rate).
- `scripts/build_notebook.py` and
  `notebooks/walk_forward_analysis.ipynb` — self-contained analysis
  notebook (executed outputs, runs against the committed CSV).
- `results/walk_forward_tech5.csv` — canonical 219-row
  walk-forward output committed for the notebook and README.
- README sections: Results (with table + four PNGs),
  `Limitations & Honest Caveats`, finbase install walkthrough.

### Changed

- Project restructured into sibling packages (`data`, `environment`,
  `evaluation`, `experiments`, `training`); `pip install -e .`
  supported.
- `PortfolioEnv.step` ~660× faster via precomputed dense numpy
  feature/price arrays; no pandas MultiIndex on the hot path.
- `prepare_for_environment` uses per-ticker `ffill`; global
  `dropna` removed (env zero-fills warm-up NaN).
- `findata` → `finbase` rename across codebase.
- `requirements.txt`: dropped `yfinance` and `polygon`.

### Fixed

- SQL injection vector in
  `PortfolioExperimentTracker.get_best_experiments`
  (whitelist-validated metric column).
- `portfolio_value <= 0` now terminates the episode with
  `portfolio_return = -1.0`.
- Bollinger Bands division-by-zero on flat-price windows (NaN
  result, handled by env `fillna`).
- Cross-ticker `ffill` pollution from naive `df.ffill()` on
  multi-index frame.
- SQLite connection leaks in `portfolio_tracker.py`
  (`contextlib.closing`).
- Defensive `.flatten()` calls on `weights` removed; 1-D float32
  invariant enforced at assignment.
- Sharpe / Sortino cold-start returns the raw step return instead
  of `0.0`.
- Monte Carlo aggregation warns on ragged-history truncation.
- NaN propagation through metrics and baselines.
- Stale `.tracker` imports in `experiments.cli` and
  `experiments.comparison`.

### Internal

- 350 unit tests passing.
- `sys.path` hacks removed.

## [0.1.0] — 2025-11-28

Initial state. Version was set in `pyproject.toml` from the initial
commit but never formally tagged.

- PPO / SAC / A2C training on `PortfolioEnv` (Gymnasium v1.0 API).
- Technical-indicator features via `pandas-ta` (RSI, MACD, BBands,
  ATR, SMA/EMA, returns, volatility, volume).
- Baseline strategies: equal-weight, buy-and-hold, momentum,
  minimum variance, inverse volatility, random.
- Stateful `Backtester`, metrics suite (Sharpe, Sortino, max
  drawdown, Calmar, win rate, profit factor).
- YAML configuration, Tensorboard logging, model checkpointing.

[0.2.0]: https://github.com/shoom1/rlportfolio/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shoom1/rlportfolio/releases/tag/v0.1.0
