"""Render a walk-forward CSV as the markdown table used in README.

The walk-forward harness emits one row per (fold, seed) with the agent's
out-of-sample Sharpe / return / drawdown / volatility on a disjoint test
window, plus the same Sharpe / return for each baseline. This script
collapses those rows into a per-strategy summary table suitable for
pasting into README under "## Results".

Usage:
    python scripts/results_table.py results/walk_forward_tech5.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Baselines tracked in the CSV. Order is the row order in the output table.
# `sp500` is added by `scripts/add_market_benchmark.py` post-hoc (the
# walk-forward harness itself doesn't run external indices).
DEFAULT_BASELINES = (
    'equal_weight', 'buy_and_hold', 'sp500',
    'momentum_20', 'min_variance_60', 'inverse_vol_30',
)


# Display labels for known columns; falls back to the column name otherwise.
DISPLAY_LABELS = {
    'sp500': 'S&P 500 (^GSPC, buy & hold)',
}


def _agg(s: pd.Series) -> dict:
    """Mean / std / median, robust to inf / NaN."""
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'median': float('nan')}
    return {'mean': float(s.mean()), 'std': float(s.std()), 'median': float(s.median())}


def render_table(csv_path: Path, baselines=DEFAULT_BASELINES) -> str:
    df = pd.read_csv(csv_path)
    if 'status' in df.columns:
        ok = df[df['status'] == 'success']
        n_fail = int((df['status'] == 'failed').sum())
    else:
        ok, n_fail = df, 0
    if ok.empty:
        return f"_(no successful runs in {csv_path.name})_"

    n_runs = len(df)
    n_ok = len(ok)
    n_folds = int(df['fold'].nunique()) if 'fold' in df.columns else n_runs
    n_seeds = int(df['seed'].nunique()) if 'seed' in df.columns else 1
    test_start = pd.to_datetime(ok['test_start']).min().date() if 'test_start' in ok.columns else '?'
    test_end = pd.to_datetime(ok['test_end']).max().date() if 'test_end' in ok.columns else '?'

    rows = []
    a_sh = _agg(ok['agent_sharpe'])
    a_ret = _agg(ok['agent_return'])
    a_dd = _agg(ok['agent_dd']) if 'agent_dd' in ok.columns else {'mean': float('nan')}
    rows.append(('**RL agent (PPO)**', a_sh, a_ret, a_dd, None))

    for b in baselines:
        if f'{b}_sharpe' not in ok.columns:
            continue
        sh = _agg(ok[f'{b}_sharpe'])
        ret = _agg(ok[f'{b}_return'])
        comparable = ok[['agent_sharpe', f'{b}_sharpe']].replace([np.inf, -np.inf], np.nan).dropna()
        hit = (comparable['agent_sharpe'] > comparable[f'{b}_sharpe']).mean() if len(comparable) else float('nan')
        label = DISPLAY_LABELS.get(b, b)
        rows.append((label, sh, ret, {'mean': float('nan')}, hit))

    lines = [
        f"_Walk-forward, expanding-window protocol. {n_folds} folds × {n_seeds} seeds "
        f"= {n_ok}/{n_runs} successful runs ({n_fail} failed). "
        f"Quarterly stride, 3-year minimum train, 6-month in-sample selection slice. "
        f"Test windows {test_start} → {test_end}._",
        "",
        "| Strategy | Mean Sharpe | Median Sharpe | Sharpe σ | Mean total return | Mean max DD | Hit rate vs agent |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, sh, ret, dd, hit in rows:
        dd_str = f"{dd['mean']:+.2%}" if dd and not np.isnan(dd['mean']) else '—'
        hit_str = '—' if hit is None else (f"{hit:.0%}" if not np.isnan(hit) else 'n/a')
        lines.append(
            f"| {name} | {sh['mean']:+.3f} | {sh['median']:+.3f} | {sh['std']:.3f} | "
            f"{ret['mean']:+.2%} | {dd_str} | {hit_str} |"
        )
    lines.append("")
    lines.append(
        "Hit rate = fraction of (fold, seed) runs where the agent's Sharpe "
        "strictly beat the baseline's on the same disjoint test window."
    )
    return "\n".join(lines)


def update_in_place(readme: Path, table: str, marker: str) -> bool:
    """Replace the block between `<!-- {marker}_START -->` and
    `<!-- {marker}_END -->` in README with `table`. Returns True if the
    README was modified."""
    text = readme.read_text()
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    if start not in text or end not in text:
        print(f"Markers {start} / {end} not found in {readme}", file=sys.stderr)
        return False
    pre, _, rest = text.partition(start)
    _, _, post = rest.partition(end)
    new = f"{pre}{start}\n{table}\n{end}{post}"
    if new == text:
        return False
    readme.write_text(new)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('csv', type=Path)
    ap.add_argument('--baselines', nargs='*', default=DEFAULT_BASELINES)
    ap.add_argument(
        '--update-readme',
        type=Path,
        help='Patch the README in-place between marker comments.',
    )
    ap.add_argument(
        '--marker',
        default='WF_TECH5_TABLE',
        help='Marker prefix to update (default: WF_TECH5_TABLE).',
    )
    args = ap.parse_args()
    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1
    table = render_table(args.csv, baselines=args.baselines)
    if args.update_readme:
        ok = update_in_place(args.update_readme, table, args.marker)
        if ok:
            print(f"Updated {args.update_readme} ({args.marker} block).")
        else:
            print(f"No changes to {args.update_readme}.", file=sys.stderr)
            return 1
    else:
        print(table)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
