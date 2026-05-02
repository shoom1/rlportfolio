"""Render walk-forward CSV results as PNGs for the README.

Produces four figures alongside the CSV:
- {prefix}_equity_curves.png — compounded equity per strategy across folds
- {prefix}_sharpe_distribution.png — per-(fold, seed) Sharpe distribution
- {prefix}_rolling_sharpe.png — rolling 4-fold mean Sharpe over time
- {prefix}_hit_rate.png — rolling agent-vs-baseline hit rate over time

Usage:
    python scripts/plot_walk_forward.py results/walk_forward_tech5.csv \
        --output-dir figures --prefix tech5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BASELINES = ('buy_and_hold', 'equal_weight', 'sp500')

# Stable colours per strategy so figures align with each other.
COLOURS = {
    'agent': '#1f77b4',          # blue
    'buy_and_hold': '#2ca02c',   # green
    'equal_weight': '#ff7f0e',   # orange
    'sp500': '#7f7f7f',          # grey
    'momentum_20': '#d62728',
    'min_variance_60': '#9467bd',
    'inverse_vol_30': '#8c564b',
}

LABELS = {
    'agent': 'PPO agent',
    'sp500': 'S&P 500 (^GSPC)',
    'buy_and_hold': 'buy_and_hold',
    'equal_weight': 'equal_weight',
}


def _label(name: str) -> str:
    return LABELS.get(name, name)


def _colour(name: str) -> str:
    return COLOURS.get(name, '#444444')


def _per_fold_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Average over seeds within each fold, NaN-safe."""
    s = df[['fold', col]].copy()
    s[col] = s[col].replace([np.inf, -np.inf], np.nan)
    return s.groupby('fold')[col].mean()


def _per_fold_dates(df: pd.DataFrame) -> pd.Series:
    """Test-window start date per fold (deduplicated; folds are
    aligned across seeds)."""
    return (
        df[['fold', 'test_start']]
        .drop_duplicates('fold')
        .set_index('fold')['test_start']
        .pipe(pd.to_datetime)
    )


def equity_curves(df: pd.DataFrame, baselines, output: Path) -> None:
    """Compound the per-fold quarterly returns into a continuous equity curve.
    Note this stitches disjoint quarters end-to-end — it is not what the
    agent would have actually earned (since each fold trains a fresh model
    on the prior history) but it shows the cumulative track record under
    the protocol."""
    fold_returns: dict[str, pd.Series] = {
        'agent': _per_fold_mean(df, 'agent_return'),
    }
    for b in baselines:
        col = f'{b}_return'
        if col in df.columns:
            fold_returns[b] = _per_fold_mean(df, col)

    dates = _per_fold_dates(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, rets in fold_returns.items():
        cum = np.cumprod(1.0 + rets.values) - 1.0
        ax.plot(dates.values, cum * 100, label=_label(name), color=_colour(name), lw=1.6)
    ax.axhline(0.0, color='black', lw=0.5, alpha=0.4)
    ax.set_title('Cumulative return — concatenated quarterly walk-forward windows')
    ax.set_ylabel('Cumulative return (%)')
    ax.set_xlabel('Test-window start')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def sharpe_distribution(df: pd.DataFrame, baselines, output: Path) -> None:
    """Box+strip plot of per-(fold, seed) Sharpe per strategy."""
    cols = ['agent_sharpe'] + [f'{b}_sharpe' for b in baselines if f'{b}_sharpe' in df.columns]
    names = ['agent'] + [b for b in baselines if f'{b}_sharpe' in df.columns]
    data = [df[c].replace([np.inf, -np.inf], np.nan).dropna().values for c in cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, vert=True, widths=0.55, patch_artist=True,
                    medianprops={'color': 'black', 'lw': 1.5},
                    boxprops={'alpha': 0.4})
    for patch, name in zip(bp['boxes'], names):
        patch.set_facecolor(_colour(name))
    # Overlay individual points (per fold-seed) with jitter.
    for i, (vals, name) in enumerate(zip(data, names), start=1):
        x = np.random.RandomState(0).normal(i, 0.06, size=len(vals))
        ax.scatter(x, vals, s=8, alpha=0.4, color=_colour(name))
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([_label(n) for n in names], rotation=15, ha='right')
    ax.axhline(0.0, color='black', lw=0.5, alpha=0.4)
    ax.set_ylabel('Sharpe (per fold × seed)')
    ax.set_title(f'Per-window Sharpe distribution ({len(data[0])} runs each)')
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def rolling_sharpe(df: pd.DataFrame, baselines, output: Path, window: int = 4) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    dates = _per_fold_dates(df)
    series_to_plot = [('agent', _per_fold_mean(df, 'agent_sharpe'))]
    for b in baselines:
        col = f'{b}_sharpe'
        if col in df.columns:
            series_to_plot.append((b, _per_fold_mean(df, col)))

    for name, sharpe_series in series_to_plot:
        s = sharpe_series.replace([np.inf, -np.inf], np.nan).rolling(window, min_periods=1).mean()
        ax.plot(dates.values, s.values, label=_label(name), color=_colour(name), lw=1.6)
    ax.axhline(0.0, color='black', lw=0.5, alpha=0.4)
    ax.set_title(f'Rolling {window}-fold mean Sharpe (≈ {window} quarters)')
    ax.set_ylabel('Mean Sharpe')
    ax.set_xlabel('Test-window start')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def hit_rate(df: pd.DataFrame, baselines, output: Path, window: int = 8) -> None:
    """Rolling fraction of (fold, seed) runs where agent_sharpe > baseline_sharpe."""
    fig, ax = plt.subplots(figsize=(10, 5))
    dates = _per_fold_dates(df)
    fold_idx = sorted(df['fold'].unique())

    for b in baselines:
        col = f'{b}_sharpe'
        if col not in df.columns:
            continue
        comp = df[['fold', 'agent_sharpe', col]].replace([np.inf, -np.inf], np.nan).dropna()
        wins_per_fold = (
            (comp['agent_sharpe'] > comp[col])
            .groupby(comp['fold']).mean()
            .reindex(fold_idx)
        )
        rolled = wins_per_fold.rolling(window, min_periods=1).mean()
        ax.plot(dates.values, rolled.values * 100, label=f'agent > {_label(b)}', color=_colour(b), lw=1.6)

    ax.axhline(50.0, color='black', lw=0.5, alpha=0.4, linestyle='--')
    ax.set_title(f'Rolling {window}-fold hit rate vs baseline (50% line = parity)')
    ax.set_ylabel('Agent wins (%)')
    ax.set_xlabel('Test-window start')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('csv', type=Path)
    ap.add_argument('--output-dir', type=Path, default=Path('figures'))
    ap.add_argument('--prefix', default=None,
                    help='Output filename prefix (default: csv stem).')
    ap.add_argument('--baselines', nargs='*', default=DEFAULT_BASELINES)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.csv.stem

    df = pd.read_csv(args.csv)
    if 'status' in df.columns:
        df = df[df['status'] == 'success'].copy()
    if df.empty:
        print(f"No successful rows in {args.csv}", file=sys.stderr)
        return 1

    baselines: List[str] = list(args.baselines)
    plots = [
        ('equity_curves', equity_curves),
        ('sharpe_distribution', sharpe_distribution),
        ('rolling_sharpe', rolling_sharpe),
        ('hit_rate', hit_rate),
    ]
    for name, fn in plots:
        out = args.output_dir / f'{prefix}_{name}.png'
        fn(df, baselines, out)
        print(f'Wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
