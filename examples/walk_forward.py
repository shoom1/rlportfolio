"""Walk-forward out-of-sample evaluation for PPO on a fixed universe.

Design:
  - Expanding training window: train on [T0, Tk - SELECTION_DAYS].
  - Model selection: last SELECTION_DAYS of each fold's train window are
    held out as an in-sample validation slice used by SB3's EvalCallback
    to save best_model. NEVER peeks at the test window.
  - Cold start: fresh PPO instance each fold.
  - Test window (tau) = stride, so test slices are disjoint quarter-by-quarter.

Produces a per-fold CSV (agent Sharpe/Return/MaxDD plus buy_and_hold and
equal_weight on the same test window) and a console aggregate (means,
hit rate, bull/bear regime split).

Runs folds in parallel across N_WORKERS processes. See the constants
block below for knobs; defaults target ~70 folds in ~30-60 min.

Note: universe, hyperparameters, and training budget are hardcoded for
the specific div19 study. Adapt the constants block for other analyses.

Usage (from project root):
    conda run -n rlportfolio python examples/walk_forward.py
"""

import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------- Config ----------------------------

TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'JPM', 'BAC', 'JNJ', 'UNH', 'XOM', 'CVX', 'PG',
    'KO', 'WMT', 'CAT', 'HON', 'LIN', 'DIS', 'VZ', 'NEE', 'SPG',
]
DATA_START = '2005-01-01'
DATA_END = '2026-12-31'

WINDOW_SIZE = 20
INITIAL_BALANCE = 10_000.0
TRANSACTION_COST = 0.001

# Fold layout (in trading days)
T_MIN_DAYS = 756         # 3 years — first fold needs at least this much train
STRIDE_DAYS = 63         # 1 quarter
TAU_DAYS = 63            # 1 quarter (matches stride -> disjoint tests)
SELECTION_DAYS = 126     # last 6 months of train reserved for EvalCallback

# Agent
TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 5_000
SEED = 42
AGENT_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
)

# Parallelism
N_WORKERS = 6

# Output
RESULTS_CSV = PROJECT_ROOT / 'results' / 'walk_forward_results.csv'


# -------- Process-wide state (populated in worker via initializer) --------

_ENV_DATA = None
_FEAT_COLS = None
_ALL_DATES = None


def _worker_init(env_data, feat_cols, all_dates):
    global _ENV_DATA, _FEAT_COLS, _ALL_DATES
    _ENV_DATA = env_data
    _FEAT_COLS = feat_cols
    _ALL_DATES = all_dates
    # Keep BLAS threads low per-worker so N_WORKERS don't oversubscribe cores.
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')


def _make_env(data_slice):
    from environment.portfolio_env import PortfolioEnv
    from environment.rewards import SharpeReward
    return PortfolioEnv(
        data=data_slice,
        feature_columns=_FEAT_COLS,
        tickers=TICKERS,
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRANSACTION_COST,
        window_size=WINDOW_SIZE,
        reward_function=SharpeReward(window=30, risk_free_rate=0.02),
    )


def _slice_by_idx(i_start, i_end):
    """Return env_data rows with dates in [_ALL_DATES[i_start], _ALL_DATES[i_end - 1]]."""
    start = _ALL_DATES[i_start]
    end = _ALL_DATES[i_end - 1]
    dates = _ENV_DATA.index.get_level_values('date')
    return _ENV_DATA[(dates >= start) & (dates <= end)]


def run_fold(fold_idx, train_start, sel_start, test_start, test_end):
    """Execute one fold end-to-end. Called in a worker process."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from evaluation import Backtester

    # Slices include WINDOW_SIZE prior days where needed so the env can
    # produce an observation at the first step of each slice.
    train_data = _slice_by_idx(train_start, sel_start)
    sel_data = _slice_by_idx(max(0, sel_start - WINDOW_SIZE), test_start)
    test_data = _slice_by_idx(max(0, test_start - WINDOW_SIZE), test_end)

    train_env = _make_env(train_data)
    sel_env = _make_env(sel_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        best_dir = Path(tmpdir) / "best"
        cb = EvalCallback(
            sel_env,
            best_model_save_path=str(best_dir),
            eval_freq=EVAL_FREQ,
            deterministic=True,
            n_eval_episodes=1,  # deterministic + fixed data -> 1 episode suffices
            render=False,
            verbose=0,
        )
        agent = PPO(
            policy='MlpPolicy',
            env=train_env,
            verbose=0,
            seed=SEED,
            **AGENT_KWARGS,
        )
        agent.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb)
        best_path = best_dir / "best_model.zip"
        # EvalCallback may not have fired (e.g., n_steps > total_timesteps) —
        # fall back to the trained agent in that case.
        model = PPO.load(str(best_path)) if best_path.exists() else agent

    bt = Backtester()
    bt.run_agent(model, _make_env(test_data), name='PPO')
    agent_m = bt.compute_metrics('PPO')

    bt.run_baseline(_make_env(test_data), 'equal_weight')
    bt.run_baseline(_make_env(test_data), 'buy_and_hold')
    bh_m = bt.compute_metrics('buy_and_hold')
    ew_m = bt.compute_metrics('equal_weight')

    return dict(
        fold=fold_idx,
        train_start=_ALL_DATES[train_start].date().isoformat(),
        sel_start=_ALL_DATES[sel_start].date().isoformat(),
        test_start=_ALL_DATES[test_start].date().isoformat(),
        test_end=_ALL_DATES[test_end - 1].date().isoformat(),
        ppo_sharpe=agent_m['sharpe_ratio'],
        ppo_return=agent_m['total_return'],
        ppo_dd=agent_m['max_drawdown'],
        ppo_vol=agent_m['volatility'],
        bh_sharpe=bh_m['sharpe_ratio'],
        bh_return=bh_m['total_return'],
        ew_sharpe=ew_m['sharpe_ratio'],
        ew_return=ew_m['total_return'],
    )


# ---------------------------- Driver ----------------------------

def prepare_data():
    from data.fetcher import DataFetcher
    from data.features import FeatureEngineer

    print(f"Fetching {TICKERS!r}\n  {DATA_START} -> {DATA_END}", flush=True)
    raw = DataFetcher().fetch_data(
        TICKERS, start_date=DATA_START, end_date=DATA_END,
    )
    engineer = FeatureEngineer()
    print("Computing features...", flush=True)
    feats = engineer.compute_features(raw)
    print("Preparing for environment...", flush=True)
    env_data = engineer.prepare_for_environment(feats)
    feat_cols = engineer.create_observation_columns(normalize=True)
    all_dates = env_data.index.get_level_values('date').unique().sort_values()
    print(f"env_data: {len(all_dates)} trading days, "
          f"{all_dates.min().date()} -> {all_dates.max().date()}", flush=True)
    return env_data, feat_cols, all_dates


def compute_folds(n_dates):
    """List of (fold_idx, train_start, sel_start, test_start, test_end) index tuples."""
    folds = []
    tk = T_MIN_DAYS
    idx = 0
    while tk + TAU_DAYS <= n_dates:
        folds.append((
            idx,                          # fold_idx
            0,                            # train_start (expanding from day 0)
            tk - SELECTION_DAYS,          # sel_start
            tk,                           # test_start
            min(tk + TAU_DAYS, n_dates),  # test_end
        ))
        tk += STRIDE_DAYS
        idx += 1
    return folds


def summarise(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print(f"AGGREGATE over {len(df)} folds")
    print("=" * 72)
    for col, label in [('ppo_sharpe', 'PPO  Sharpe'),
                       ('bh_sharpe',  'BH   Sharpe'),
                       ('ew_sharpe',  'EW   Sharpe')]:
        s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"{label}: mean={s.mean():+.3f}  std={s.std():.3f}  "
              f"median={s.median():+.3f}  min={s.min():+.3f}  max={s.max():+.3f}")

    ppo_beats_bh = (df['ppo_sharpe'] > df['bh_sharpe']).mean()
    ppo_beats_ew = (df['ppo_sharpe'] > df['ew_sharpe']).mean()
    print(f"\nPPO Sharpe > BH Sharpe: {ppo_beats_bh:.1%} of folds")
    print(f"PPO Sharpe > EW Sharpe: {ppo_beats_ew:.1%} of folds")

    bull = df[df['bh_return'] > 0]
    bear = df[df['bh_return'] <= 0]
    print(f"\nBull folds (buy_and_hold return > 0): {len(bull)}")
    if len(bull):
        print(f"  PPO Sharpe median: {bull['ppo_sharpe'].median():+.3f}  "
              f"BH median: {bull['bh_sharpe'].median():+.3f}  "
              f"EW median: {bull['ew_sharpe'].median():+.3f}")
    print(f"Bear folds (buy_and_hold return <= 0): {len(bear)}")
    if len(bear):
        print(f"  PPO Sharpe median: {bear['ppo_sharpe'].median():+.3f}  "
              f"BH median: {bear['bh_sharpe'].median():+.3f}  "
              f"EW median: {bear['ew_sharpe'].median():+.3f}")

    print(f"\nPPO Max DD: median={df['ppo_dd'].median():+.2%}  "
          f"min={df['ppo_dd'].min():+.2%}  max={df['ppo_dd'].max():+.2%}")


def main():
    env_data, feat_cols, all_dates = prepare_data()
    folds = compute_folds(len(all_dates))
    print(f"\n{len(folds)} folds: "
          f"first test {all_dates[T_MIN_DAYS].date()}, "
          f"last test ends {all_dates[folds[-1][4]-1].date()}", flush=True)
    print(f"Launching {N_WORKERS} parallel workers...\n", flush=True)

    rows = []
    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_worker_init,
        initargs=(env_data, feat_cols, all_dates),
    ) as ex:
        futures = {ex.submit(run_fold, *fold): fold[0] for fold in folds}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                rows.append(r)
                print(
                    f"[fold {r['fold']:3d}] {r['test_start']} -> {r['test_end']}  "
                    f"PPO Sh={r['ppo_sharpe']:+.2f} R={r['ppo_return']:+.2%}  "
                    f"BH Sh={r['bh_sharpe']:+.2f}  EW Sh={r['ew_sharpe']:+.2f}",
                    flush=True,
                )
            except Exception as e:
                fold_idx = futures[fut]
                print(f"[fold {fold_idx}] FAILED: {e!r}", flush=True)

    df = pd.DataFrame(rows).sort_values('fold').reset_index(drop=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults written to {RESULTS_CSV}", flush=True)

    summarise(df)


if __name__ == "__main__":
    main()
