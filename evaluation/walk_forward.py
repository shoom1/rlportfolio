"""Walk-forward out-of-sample evaluation harness.

Trains a fresh SB3 agent on each fold's expanding window and backtests it
on the next disjoint test slice. Each fold can be repeated across multiple
training seeds. The last `selection_days` of each fold's train window are
held out as an in-sample validation set used by `EvalCallback` to save
`best_model`. The test window is **never** touched during training or model
selection.

Complements `WalkForwardBacktestStrategy` in `evaluation.backtest_strategies`:
that class rolls a pre-trained agent through windows without retraining,
which is a different use case.

Programmatic use::

    from training.config import TrainingConfig
    from evaluation.walk_forward import WalkForwardEvaluator, WalkForwardConfig

    cfg = TrainingConfig.from_yaml("configs/opt_c_div19.yaml")
    ev = WalkForwardEvaluator(cfg=cfg, wf=WalkForwardConfig())
    df = ev.run()
    stats = WalkForwardEvaluator.summarise(df, baselines=ev.wf.baselines)

CLI use::

    python -m evaluation.walk_forward --config configs/opt_c_div19.yaml \\
        --t-min-days 756 --stride-days 63 \\
        --seeds 42 43 44 \\
        --output results/walk_forward.csv
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.fetcher import DataFetcher
from data.features import FeatureEngineer
from environment.portfolio_env import PortfolioEnv
from environment.rewards import RewardFunctionFactory
from training.config import TrainingConfig

from .backtest import Backtester
from .baselines import BaselineStrategyRegistry


# Tuple layout emitted by compute_folds and consumed by the worker.
FoldSpec = Tuple[int, int, int, int, int]
# (fold_idx, train_start, sel_start, test_start, test_end) — all are positional
# indices into the sorted unique-dates array.


def _algorithm_class(name: str):
    """Resolve an SB3 algorithm name to its class via the registry."""
    # Imported here to avoid pulling training at module-import time.
    from training.train import PortfolioTrainer
    if name not in PortfolioTrainer.ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm {name!r}; available: "
            f"{sorted(PortfolioTrainer.ALGORITHM_REGISTRY)}"
        )
    return PortfolioTrainer.ALGORITHM_REGISTRY[name]


@dataclass
class WalkForwardConfig:
    """Fold geometry and parallelism knobs.

    All counts are in trading days (not calendar days). Defaults match the
    quarterly protocol used in the div19 study: 3-year minimum train, 1-quarter
    stride, 1-quarter disjoint test, 6-month in-sample model-selection slice.
    """

    t_min_days: int = 756          # first fold needs at least this much train
    stride_days: int = 63          # 1 quarter
    tau_days: int = 63             # test window; disjoint when == stride_days
    selection_days: int = 126      # in-sample val for EvalCallback
    n_workers: int = 6
    seed: int = 42
    seeds: Optional[List[int]] = None
    baselines: List[str] = field(
        default_factory=lambda: ['buy_and_hold', 'equal_weight']
    )

    def __post_init__(self) -> None:
        if self.t_min_days <= 0:
            raise ValueError(f"t_min_days must be > 0, got {self.t_min_days}")
        if self.stride_days <= 0:
            raise ValueError(f"stride_days must be > 0, got {self.stride_days}")
        if self.tau_days <= 0:
            raise ValueError(f"tau_days must be > 0, got {self.tau_days}")
        if self.selection_days < 0:
            raise ValueError(
                f"selection_days must be >= 0, got {self.selection_days}"
            )
        if self.n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {self.n_workers}")
        if self.seeds is not None:
            if len(self.seeds) == 0:
                raise ValueError("seeds must contain at least one seed")
            self.seeds = [int(seed) for seed in self.seeds]
            if len(set(self.seeds)) != len(self.seeds):
                raise ValueError(f"seeds must be unique, got {self.seeds}")
        if self.t_min_days <= self.selection_days:
            raise ValueError(
                f"t_min_days ({self.t_min_days}) must exceed selection_days "
                f"({self.selection_days}) — otherwise the first fold has no "
                f"training data outside the selection slice"
            )
        known = set(BaselineStrategyRegistry().list_strategies())
        unknown = [b for b in self.baselines if b not in known]
        if unknown:
            raise ValueError(
                f"Unknown baselines: {unknown}. "
                f"Available: {sorted(known)}"
            )

    def seed_values(self) -> List[int]:
        """Training seeds to run per fold."""
        return [self.seed] if self.seeds is None else list(self.seeds)


# --------- Module-level state used by worker processes -----------
# ProcessPoolExecutor spawns workers via `initializer` once; subsequent
# submit() calls then need only pickle the fold tuple, not the full
# env_data / config. These globals are set by _worker_init per worker.

_ENV_DATA: Optional[pd.DataFrame] = None
_FEAT_COLS: Optional[List[str]] = None
_ALL_DATES: Optional[pd.DatetimeIndex] = None
_CFG: Optional[TrainingConfig] = None
_WF: Optional[WalkForwardConfig] = None


def _worker_init(env_data, feat_cols, all_dates, cfg, wf) -> None:
    global _ENV_DATA, _FEAT_COLS, _ALL_DATES, _CFG, _WF
    _ENV_DATA = env_data
    _FEAT_COLS = feat_cols
    _ALL_DATES = all_dates
    _CFG = cfg
    _WF = wf
    # Keep per-worker BLAS threads low so N workers don't oversubscribe cores.
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')


def _make_env(data_slice: pd.DataFrame) -> PortfolioEnv:
    reward = RewardFunctionFactory.create(
        _CFG.environment.reward_function,
        **_CFG.environment.reward_params,
    )
    return PortfolioEnv(
        data=data_slice,
        feature_columns=_FEAT_COLS,
        tickers=_CFG.data.tickers,
        initial_balance=_CFG.environment.initial_balance,
        transaction_cost=_CFG.environment.transaction_cost,
        window_size=_CFG.environment.window_size,
        reward_function=reward,
    )


def _slice_by_idx(i_start: int, i_end: int) -> pd.DataFrame:
    """Rows with date in [_ALL_DATES[i_start], _ALL_DATES[i_end - 1]]."""
    start = _ALL_DATES[i_start]
    end = _ALL_DATES[i_end - 1]
    dates = _ENV_DATA.index.get_level_values('date')
    return _ENV_DATA[(dates >= start) & (dates <= end)]


def _date_iso(all_dates: pd.DatetimeIndex, idx: int) -> str:
    return pd.Timestamp(all_dates[idx]).date().isoformat()


def _fold_metadata_row(
    fold: FoldSpec,
    all_dates: pd.DatetimeIndex,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    fold_idx, train_start, sel_start, test_start, test_end = fold
    row = dict(
        fold=fold_idx,
        train_start=_date_iso(all_dates, train_start),
        sel_start=_date_iso(all_dates, sel_start),
        test_start=_date_iso(all_dates, test_start),
        test_end=_date_iso(all_dates, test_end - 1),
    )
    if seed is not None:
        row['seed'] = seed
    return row


def _failed_fold_row(
    fold: FoldSpec,
    all_dates: pd.DatetimeIndex,
    exc: BaseException,
    baselines: List[str],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    row = _fold_metadata_row(fold, all_dates, seed=seed)
    row.update(
        status='failed',
        agent_sharpe=np.nan,
        agent_return=np.nan,
        agent_dd=np.nan,
        agent_vol=np.nan,
        error_type=type(exc).__name__,
        error_message=str(exc),
        error_traceback=''.join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    )
    for b in baselines:
        row[f'{b}_sharpe'] = np.nan
        row[f'{b}_return'] = np.nan
    return row


def _run_fold_worker(
    fold_idx: int,
    train_start: int,
    sel_start: int,
    test_start: int,
    test_end: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Train+test one fold. Module-level so it's picklable by spawn workers."""
    from stable_baselines3.common.callbacks import EvalCallback

    run_seed = _WF.seed if seed is None else int(seed)
    window = _CFG.environment.window_size
    # Include `window` prior days in sel/test slices so the env can emit an
    # observation at step 0 of each slice.
    train_data = _slice_by_idx(train_start, sel_start)
    sel_data = _slice_by_idx(max(0, sel_start - window), test_start)
    test_data = _slice_by_idx(max(0, test_start - window), test_end)

    algo_cls = _algorithm_class(_CFG.agent.algorithm)

    with tempfile.TemporaryDirectory() as tmpdir:
        best_dir = Path(tmpdir) / "best"
        cb = EvalCallback(
            _make_env(sel_data),
            best_model_save_path=str(best_dir),
            eval_freq=_CFG.training.eval_freq,
            deterministic=True,
            n_eval_episodes=1,
            render=False,
            verbose=0,
        )
        agent = algo_cls(
            policy=_CFG.agent.policy,
            env=_make_env(train_data),
            verbose=0,
            seed=run_seed,
            **_CFG.agent.extra,
        )
        agent.learn(
            total_timesteps=_CFG.training.total_timesteps,
            callback=cb,
        )
        best_path = best_dir / "best_model.zip"
        # EvalCallback may not have fired (e.g., n_steps > total_timesteps) —
        # fall back to the trained agent in that case.
        model = algo_cls.load(str(best_path)) if best_path.exists() else agent

    bt = Backtester()
    bt.run_agent(model, _make_env(test_data), name='agent')
    agent_m = bt.compute_metrics('agent')

    row: Dict[str, Any] = _fold_metadata_row(
        (fold_idx, train_start, sel_start, test_start, test_end),
        _ALL_DATES,
        seed=run_seed,
    )
    row.update(
        status='success',
        agent_sharpe=agent_m['sharpe_ratio'],
        agent_return=agent_m['total_return'],
        agent_dd=agent_m['max_drawdown'],
        agent_vol=agent_m['volatility'],
        error_type='',
        error_message='',
        error_traceback='',
    )
    for b in _WF.baselines:
        bt.run_baseline(_make_env(test_data), b)
        m = bt.compute_metrics(b)
        row[f'{b}_sharpe'] = m['sharpe_ratio']
        row[f'{b}_return'] = m['total_return']
    return row


class WalkForwardEvaluator:
    """Expanding-window walk-forward training + OOS backtesting.

    Prepares env data once, then dispatches one training+backtest job per
    fold/seed pair across a pool of processes. Returns a per-run DataFrame
    containing the agent's Sharpe/return/DD/vol on its disjoint test window,
    plus the same metrics for each requested baseline on that same window.

    See the module docstring for how train / model-selection / test windows
    are laid out.
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        wf: Optional[WalkForwardConfig] = None,
        data_start: str = '2005-01-01',
        data_end: str = '2026-12-31',
        data_fetcher=None,
        feature_engineer=None,
    ) -> None:
        self.cfg = cfg
        self.wf = wf or WalkForwardConfig()
        self.data_start = data_start
        self.data_end = data_end
        self.data_fetcher = data_fetcher or DataFetcher()
        self.feature_engineer = feature_engineer or FeatureEngineer()

    def prepare_data(self):
        """Fetch, compute features, and return (env_data, feat_cols, all_dates).

        Feature normalisation in `prepare_for_environment` is causal (rolling
        `pct_change`, scalar divisions), so fetching once and slicing per fold
        does NOT introduce look-ahead.
        """
        print(f"Fetching {self.cfg.data.tickers}\n"
              f"  {self.data_start} -> {self.data_end}", flush=True)
        raw = self.data_fetcher.fetch_data(
            self.cfg.data.tickers,
            start_date=self.data_start,
            end_date=self.data_end,
        )
        print("Computing features...", flush=True)
        feats = self.feature_engineer.compute_features(raw)
        print("Preparing for environment...", flush=True)
        env_data = self.feature_engineer.prepare_for_environment(feats)
        feat_cols = self.feature_engineer.create_observation_columns(normalize=True)
        all_dates = env_data.index.get_level_values('date').unique().sort_values()
        print(
            f"env_data: {len(all_dates)} trading days, "
            f"{all_dates.min().date()} -> {all_dates.max().date()}",
            flush=True,
        )
        return env_data, feat_cols, all_dates

    @staticmethod
    def compute_folds(wf: WalkForwardConfig, n_dates: int) -> List[FoldSpec]:
        """Fold index tuples (fold_idx, train_start, sel_start, test_start, test_end).

        Walk-forward layout with expanding training window, quarterly stride
        by default. Pure function — easy to unit-test.
        """
        folds: List[FoldSpec] = []
        tk = wf.t_min_days
        idx = 0
        while tk + wf.tau_days <= n_dates:
            folds.append((
                idx,
                0,                                # train always expands from day 0
                tk - wf.selection_days,           # sel_start
                tk,                                # test_start
                min(tk + wf.tau_days, n_dates),   # test_end
            ))
            tk += wf.stride_days
            idx += 1
        return folds

    def run(self) -> pd.DataFrame:
        """Run all fold/seed jobs in parallel; return a sorted DataFrame."""
        env_data, feat_cols, all_dates = self.prepare_data()
        folds = self.compute_folds(self.wf, len(all_dates))
        seeds = self.wf.seed_values()
        if not folds:
            raise RuntimeError(
                f"No folds fit in the available data: need at least "
                f"{self.wf.t_min_days + self.wf.tau_days} trading days, "
                f"got {len(all_dates)}"
            )
        print(
            f"\n{len(folds)} folds x {len(seeds)} seeds = "
            f"{len(folds) * len(seeds)} runs: "
            f"first test {all_dates[self.wf.t_min_days].date()}, "
            f"last test ends {all_dates[folds[-1][4] - 1].date()}",
            flush=True,
        )
        print(f"Launching {self.wf.n_workers} parallel workers...\n", flush=True)

        rows: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(
            max_workers=self.wf.n_workers,
            initializer=_worker_init,
            initargs=(env_data, feat_cols, all_dates, self.cfg, self.wf),
        ) as ex:
            futures = {
                ex.submit(_run_fold_worker, *fold, seed): (fold, seed)
                for fold in folds
                for seed in seeds
            }
            for fut in as_completed(futures):
                fold, seed = futures[fut]
                fold_idx = fold[0]
                try:
                    r = fut.result()
                except Exception as e:
                    print(
                        f"[fold {fold_idx} seed {seed}] FAILED: {e!r}",
                        flush=True,
                    )
                    rows.append(
                        _failed_fold_row(
                            fold,
                            all_dates,
                            e,
                            self.wf.baselines,
                            seed=seed,
                        )
                    )
                    continue
                rows.append(r)
                base_strs = "  ".join(
                    f"{b} Sh={r[f'{b}_sharpe']:+.2f}"
                    for b in self.wf.baselines
                )
                print(
                    f"[fold {r['fold']:3d} seed {r['seed']}] "
                    f"{r['test_start']} -> {r['test_end']}  "
                    f"agent Sh={r['agent_sharpe']:+.2f} R={r['agent_return']:+.2%}  "
                    f"{base_strs}",
                    flush=True,
                )

        sort_cols = ['fold', 'seed'] if rows and 'seed' in rows[0] else ['fold']
        return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)

    @staticmethod
    def summarise(df: pd.DataFrame, baselines: List[str]) -> Dict[str, Any]:
        """Aggregate mean / std / median / min / max Sharpe and hit rate."""
        if 'status' in df.columns:
            successful = df[df['status'] == 'success']
            n_failed = int((df['status'] == 'failed').sum())
        else:
            successful = df
            n_failed = 0

        def _agg_series(s: pd.Series) -> Dict[str, float]:
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            return {
                'mean': float(s.mean()),
                'std': float(s.std()),
                'median': float(s.median()),
                'min': float(s.min()),
                'max': float(s.max()),
            }

        def _agg(col: str) -> Dict[str, float]:
            return _agg_series(successful[col])

        def _fold_mean_agg(col: str) -> Dict[str, float]:
            if 'fold' not in successful.columns:
                return _agg(col)
            s = successful[col].replace([np.inf, -np.inf], np.nan)
            return _agg_series(s.groupby(successful['fold']).mean())

        n_runs = len(df)
        n_successful_runs = len(successful)
        n_folds = int(df['fold'].nunique()) if 'fold' in df.columns else n_runs
        n_successful_folds = (
            int(successful['fold'].nunique())
            if 'fold' in successful.columns
            else n_successful_runs
        )
        n_failed_folds = (
            int(df.loc[df.get('status', '') == 'failed', 'fold'].nunique())
            if 'fold' in df.columns and 'status' in df.columns
            else n_failed
        )

        out: Dict[str, Any] = {
            'n_runs': n_runs,
            'n_successful_runs': n_successful_runs,
            'n_failed_runs': n_failed,
            'n_folds': n_folds,
            'n_successful_folds': n_successful_folds,
            'n_failed_folds': n_failed_folds,
            'agent_sharpe': _agg('agent_sharpe'),
            'agent_sharpe_by_fold': _fold_mean_agg('agent_sharpe'),
        }
        for b in baselines:
            col = f'{b}_sharpe'
            if col not in df.columns:
                continue
            out[f'{b}_sharpe'] = _agg(col)
            out[f'{b}_sharpe_by_fold'] = _fold_mean_agg(col)
            comparable = (
                successful[['agent_sharpe', col]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            hit_rate = (
                (comparable['agent_sharpe'] > comparable[col]).mean()
                if len(comparable) > 0
                else np.nan
            )
            out[f'{b}_hit_rate'] = float(hit_rate)
        return out


# ------------------------------ CLI ---------------------------------

def _format_summary(df: pd.DataFrame, stats: Dict[str, Any], baselines: List[str]) -> str:
    n_runs = stats.get('n_runs', stats['n_folds'])
    n_successful_runs = stats.get('n_successful_runs', stats['n_successful_folds'])
    n_failed_runs = stats.get('n_failed_runs', stats.get('n_failed_folds', 0))
    lines = [
        "=" * 72,
        f"AGGREGATE over {n_successful_runs}/{n_runs} successful runs "
        f"across {stats['n_folds']} folds",
        "=" * 72,
    ]
    if n_failed_runs:
        lines.append(
            f"failed runs recorded: {n_failed_runs} "
            f"across {stats.get('n_failed_folds', n_failed_runs)} folds"
        )
    a = stats['agent_sharpe']
    lines.append(
        f"agent Sharpe: mean={a['mean']:+.3f}  std={a['std']:.3f}  "
        f"median={a['median']:+.3f}  min={a['min']:+.3f}  max={a['max']:+.3f}"
    )
    if 'agent_sharpe_by_fold' in stats:
        af = stats['agent_sharpe_by_fold']
        lines.append(
            f"agent fold-mean Sharpe: mean={af['mean']:+.3f}  std={af['std']:.3f}  "
            f"median={af['median']:+.3f}"
        )
    for b in baselines:
        key = f'{b}_sharpe'
        if key not in stats:
            continue
        s = stats[key]
        lines.append(
            f"{b:>20s} Sharpe: mean={s['mean']:+.3f}  median={s['median']:+.3f}   "
            f"agent > {b}: {stats[f'{b}_hit_rate']:.1%} of successful runs"
        )
    return "\n".join(lines)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk-forward OOS evaluation of an SB3 agent on PortfolioEnv."
    )
    parser.add_argument('--config', type=Path, required=True,
                        help="Path to a TrainingConfig YAML.")
    parser.add_argument('--data-start', type=str, default='2005-01-01')
    parser.add_argument('--data-end', type=str, default='2026-12-31')
    parser.add_argument('--t-min-days', type=int, default=756,
                        help="Min train length before first fold (trading days).")
    parser.add_argument('--stride-days', type=int, default=63,
                        help="Fold stride in trading days (default: 63 = 1 quarter).")
    parser.add_argument('--tau-days', type=int, default=None,
                        help="Test window length; defaults to --stride-days (disjoint tests).")
    parser.add_argument('--selection-days', type=int, default=126,
                        help="Tail of train reserved for EvalCallback (default: 126 = 6 months).")
    parser.add_argument('--n-workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42,
                        help="Single training seed used when --seeds is omitted.")
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help="Optional list of training seeds to run per fold.")
    parser.add_argument('--baselines', nargs='+',
                        default=['buy_and_hold', 'equal_weight'])
    parser.add_argument('--output', type=Path, default=None,
                        help="CSV path for per-fold results. "
                             "Default: walk_forward_<config-stem>.csv in cwd.")
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)
    wf = WalkForwardConfig(
        t_min_days=args.t_min_days,
        stride_days=args.stride_days,
        tau_days=args.tau_days if args.tau_days is not None else args.stride_days,
        selection_days=args.selection_days,
        n_workers=args.n_workers,
        seed=args.seed,
        seeds=args.seeds,
        baselines=args.baselines,
    )
    evaluator = WalkForwardEvaluator(
        cfg=cfg, wf=wf,
        data_start=args.data_start, data_end=args.data_end,
    )

    df = evaluator.run()

    output = args.output or Path(f'walk_forward_{args.config.stem}.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"\nResults written to {output}", flush=True)

    stats = WalkForwardEvaluator.summarise(df, wf.baselines)
    print("\n" + _format_summary(df, stats, wf.baselines))


if __name__ == "__main__":
    _main()
