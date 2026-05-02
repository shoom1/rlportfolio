"""Append a market-index benchmark (default: SPY) to a walk-forward CSV.

The walk-forward harness in `evaluation.walk_forward` runs the agent and
in-universe baselines through `PortfolioEnv` and writes per-fold metrics
to CSV. It does NOT include a broad-market index (e.g. SPY) because that
ticker is typically not in the configured universe — pulling it in would
require either special-casing or duplicating the env.

This script fetches the index separately, computes buy-and-hold Sharpe
and total return on each (test_start, test_end) window in the CSV using
the same `PortfolioMetrics` machinery as the harness, applies a single
initial transaction cost, and writes the columns back to the CSV
in-place. Running it twice on the same CSV is idempotent.

Usage:
    python scripts/add_market_benchmark.py results/walk_forward_tech5.csv
    python scripts/add_market_benchmark.py results/walk_forward_tech5.csv --ticker QQQ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.fetcher import DataFetcher
from evaluation.metrics import PortfolioMetrics


# Match the env's default proportional cost (configs/opt_c_tech5.yaml).
DEFAULT_INITIAL_COST = 0.001


def _fetch_index_closes(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return a date-indexed Series of close prices for `ticker` covering
    [start, end] inclusive. Falls back to a wider window because finbase's
    range may not start exactly on `start` (weekends/holidays/IPO dates)."""
    fetcher = DataFetcher()
    pad = pd.Timedelta(days=10)
    df = fetcher.fetch_data(
        [ticker],
        start_date=(start - pad).date().isoformat(),
        end_date=(end + pad).date().isoformat(),
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    closes = df.xs(ticker, level='ticker')['close'].sort_index()
    return closes


def _window_metrics(
    closes: pd.Series,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    initial_cost: float,
) -> dict:
    """Compute buy-and-hold Sharpe and total_return for `ticker` between
    `test_start` and `test_end` (both inclusive, calendar days). The
    window is clipped to the actual trading bars available."""
    window = closes[(closes.index >= test_start) & (closes.index <= test_end)]
    if len(window) < 2:
        return {'sharpe': float('nan'), 'return': float('nan')}

    rets = window.pct_change().dropna()
    # Buy-and-hold: one initial buy of the index, no further trades.
    # Subtract initial cost from the first daily return so total_return
    # and Sharpe both reflect it (matches what PortfolioEnv would do
    # through its proportional cost model on a single rebalance).
    if len(rets) > 0:
        rets = rets.copy()
        rets.iloc[0] -= initial_cost

    portfolio_value = float(window.iloc[0]) * np.cumprod(1.0 + rets.values)
    history = pd.DataFrame({
        'date': rets.index,
        'value': portfolio_value,
        'return': rets.values,
    })

    metrics = PortfolioMetrics().calculate_all_metrics(history)
    return {'sharpe': float(metrics['sharpe_ratio']), 'return': float(metrics['total_return'])}


def add_benchmark(
    csv_path: Path,
    ticker: str = 'SPY',
    initial_cost: float = DEFAULT_INITIAL_COST,
    column_prefix: str | None = None,
) -> int:
    """Append `<prefix>_sharpe` and `<prefix>_return` columns to the CSV."""
    prefix = column_prefix or ticker.lower()
    df = pd.read_csv(csv_path)

    if 'test_start' not in df.columns or 'test_end' not in df.columns:
        raise ValueError(f"{csv_path} missing test_start/test_end columns")

    df['test_start'] = pd.to_datetime(df['test_start'])
    df['test_end'] = pd.to_datetime(df['test_end'])

    overall_start = df['test_start'].min()
    overall_end = df['test_end'].max()
    print(f"Fetching {ticker} from {overall_start.date()} to {overall_end.date()}...")
    closes = _fetch_index_closes(ticker, overall_start, overall_end)
    print(f"  {len(closes)} trading bars")

    # Group by unique (test_start, test_end) to avoid recomputing per seed.
    windows = df[['test_start', 'test_end']].drop_duplicates()
    print(f"Computing buy-and-hold {ticker} on {len(windows)} unique test windows...")
    per_window: dict[tuple, dict] = {}
    for ts, te in windows.itertuples(index=False):
        per_window[(ts, te)] = _window_metrics(closes, ts, te, initial_cost)

    df[f'{prefix}_sharpe'] = df.apply(
        lambda r: per_window[(r['test_start'], r['test_end'])]['sharpe'], axis=1,
    )
    df[f'{prefix}_return'] = df.apply(
        lambda r: per_window[(r['test_start'], r['test_end'])]['return'], axis=1,
    )

    # Restore the date columns to ISO-string form so the file round-trips
    # cleanly through `results_table.py` and downstream consumers.
    df['test_start'] = df['test_start'].dt.date.astype(str)
    df['test_end'] = df['test_end'].dt.date.astype(str)

    df.to_csv(csv_path, index=False)
    finite = df[f'{prefix}_sharpe'].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Wrote {prefix}_sharpe / {prefix}_return for {len(finite)} rows.")
    print(f"  mean Sharpe: {finite.mean():+.3f}  median: {finite.median():+.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('csv', type=Path)
    ap.add_argument('--ticker', default='SPY', help='Index ticker (default: SPY).')
    ap.add_argument(
        '--initial-cost', type=float, default=DEFAULT_INITIAL_COST,
        help='Proportional cost applied once at the start of each window '
             '(default: 0.001 = 10 bps, matching the env).',
    )
    ap.add_argument(
        '--column-prefix', default=None,
        help='CSV column prefix (default: lowercase ticker).',
    )
    args = ap.parse_args()
    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1
    return add_benchmark(args.csv, args.ticker, args.initial_cost, args.column_prefix)


if __name__ == '__main__':
    raise SystemExit(main())
