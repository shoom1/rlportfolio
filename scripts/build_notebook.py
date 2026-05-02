"""Generate `notebooks/walk_forward_analysis.ipynb` from this script.

The notebook is the source-of-truth for visual analysis of a walk-forward
CSV. Storing it as JSON in git makes it noisy to diff and risky to edit
by hand, so we keep the canonical content here in code and re-emit the
notebook from this generator. To regenerate after changes:

    conda run -n rlportfolio python scripts/build_notebook.py

The notebook is self-contained: it loads the committed
`results/walk_forward_tech5.csv` (kept in git via a `.gitignore`
exception) and uses only matplotlib + pandas + numpy, all of which are
already in `environment.yml` / `requirements.txt`. No finbase access,
no SB3, no training required to run the notebook.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / 'notebooks' / 'walk_forward_analysis.ipynb'
CSV_REL_PATH = '../results/walk_forward_tech5.csv'


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip() + '\n')


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3 (rlportfolio)',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {'name': 'python'},
    }

    cells = []

    cells.append(md("""
        # Walk-forward analysis — tech5 universe

        This notebook is **self-contained**: it reads the committed CSV
        `results/walk_forward_tech5.csv` and reproduces the figures embedded in
        the project README, plus a few extra cells that are easier to explore
        interactively than as static PNGs.

        The CSV was produced by:

        ```bash
        conda run -n rlportfolio python -m evaluation.walk_forward \\
            --config configs/opt_c_tech5.yaml \\
            --t-min-days 756 --stride-days 63 \\
            --seeds 42 43 44 \\
            --output results/walk_forward_tech5.csv

        conda run -n rlportfolio python scripts/add_market_benchmark.py \\
            results/walk_forward_tech5.csv \\
            --ticker '^GSPC' --column-prefix sp500
        ```

        You do **not** need finbase, stable-baselines3, or any market data
        access to run this notebook — only `pandas`, `numpy`, and
        `matplotlib`, all of which ship with `environment.yml`.
    """))

    cells.append(md("## 1. Load the CSV and inspect schema"))

    cells.append(code(f"""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path

        plt.rcParams['figure.dpi'] = 110
        pd.set_option('display.max_rows', 20)

        CSV = Path('{CSV_REL_PATH}')
        df = pd.read_csv(CSV)
        df['test_start'] = pd.to_datetime(df['test_start'])
        df['test_end'] = pd.to_datetime(df['test_end'])

        # Drop failed runs from analysis (none in the committed CSV, but kept
        # for safety on future re-runs).
        if 'status' in df.columns:
            df = df[df['status'] == 'success'].copy()

        print(f'rows                : {{len(df):4d}}')
        print(f'unique folds        : {{df["fold"].nunique():4d}}')
        print(f'unique seeds        : {{df["seed"].nunique():4d}}')
        print(f'first test window   : {{df["test_start"].min().date()}}')
        print(f'last test window    : {{df["test_end"].max().date()}}')
        print()
        df.head()
    """))

    cells.append(md("## 2. Aggregate summary"))

    cells.append(code("""
        BASELINES = ['buy_and_hold', 'equal_weight', 'sp500']

        rows = []
        for name, sharpe_col, ret_col in [
            ('PPO agent', 'agent_sharpe', 'agent_return'),
            *[(b, f'{b}_sharpe', f'{b}_return') for b in BASELINES],
        ]:
            sh = df[sharpe_col].replace([np.inf, -np.inf], np.nan).dropna()
            rt = df[ret_col].replace([np.inf, -np.inf], np.nan).dropna()
            if name == 'PPO agent':
                hit = ''
            else:
                comp = df[['agent_sharpe', sharpe_col]].replace([np.inf, -np.inf], np.nan).dropna()
                hit = f'{(comp.agent_sharpe > comp[sharpe_col]).mean():.0%}' if len(comp) else 'n/a'
            rows.append({
                'strategy': name,
                'mean_sharpe': sh.mean(),
                'median_sharpe': sh.median(),
                'sharpe_std': sh.std(),
                'mean_return': rt.mean(),
                'hit_rate_vs_agent': hit,
            })

        summary = pd.DataFrame(rows)
        summary
    """))

    cells.append(md("""
        On the tech5 universe, with these features and this training budget,
        the agent's **mean Sharpe trails the in-universe baselines by ~0.07**
        and beats them in only 40 % of (fold, seed) runs. The agent edges the
        S&P 500 — but so do the in-universe baselines by an even larger
        margin, so the apparent edge over the broad market is **universe
        selection, not the RL policy**.
    """))

    cells.append(md("## 3. Cumulative return over the full walk-forward path"))

    cells.append(md("""
        Each fold reports a per-quarter total return. Stitching them
        end-to-end gives a continuous "what would $1 have grown to under this
        protocol" curve. Note this is *not* an actual tradable equity curve
        because each fold's agent is a fresh model trained on the prior
        history — but it is the right comparison for the protocol because
        every strategy gets the same treatment.
    """))

    cells.append(code("""
        def per_fold_mean(col: str) -> pd.Series:
            return (df[['fold', col]]
                    .replace([np.inf, -np.inf], np.nan)
                    .groupby('fold')[col].mean())

        dates = (df[['fold', 'test_start']]
                 .drop_duplicates('fold')
                 .set_index('fold')['test_start'])

        fig, ax = plt.subplots(figsize=(10, 5))
        for col, label, colour in [
            ('agent_return',         'PPO agent',       '#1f77b4'),
            ('buy_and_hold_return',  'buy_and_hold',    '#2ca02c'),
            ('equal_weight_return',  'equal_weight',    '#ff7f0e'),
            ('sp500_return',         'S&P 500 (^GSPC)', '#7f7f7f'),
        ]:
            cum = np.cumprod(1.0 + per_fold_mean(col).values) - 1.0
            ax.plot(dates.values, cum * 100, label=label, color=colour, lw=1.6)
        ax.axhline(0, color='black', lw=0.5, alpha=0.4)
        ax.set_title('Cumulative return — concatenated quarterly walk-forward windows')
        ax.set_ylabel('Cumulative return (%)')
        ax.set_xlabel('Test-window start')
        ax.legend(loc='upper left', frameon=False)
        ax.grid(alpha=0.25)
        plt.show()
    """))

    cells.append(md("## 4. Per-window Sharpe distribution"))

    cells.append(md("""
        Boxplots collapse the 219 per-(fold, seed) Sharpe values per strategy.
        Heavy overlap is the headline: at the resolution of a single quarterly
        window, the four distributions are nearly indistinguishable.
    """))

    cells.append(code("""
        cols = ['agent_sharpe'] + [f'{b}_sharpe' for b in BASELINES]
        labels = ['PPO agent', 'buy_and_hold', 'equal_weight', 'S&P 500 (^GSPC)']
        colours = ['#1f77b4', '#2ca02c', '#ff7f0e', '#7f7f7f']
        data = [df[c].replace([np.inf, -np.inf], np.nan).dropna().values for c in cols]

        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(data, vert=True, widths=0.55, patch_artist=True,
                        medianprops={'color': 'black', 'lw': 1.5},
                        boxprops={'alpha': 0.4})
        for patch, c in zip(bp['boxes'], colours):
            patch.set_facecolor(c)
        rng = np.random.RandomState(0)
        for i, (vals, c) in enumerate(zip(data, colours), start=1):
            ax.scatter(rng.normal(i, 0.06, size=len(vals)), vals, s=8, alpha=0.4, color=c)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.axhline(0, color='black', lw=0.5, alpha=0.4)
        ax.set_ylabel('Sharpe (per fold × seed)')
        ax.set_title(f'Per-window Sharpe distribution ({len(data[0])} runs each)')
        ax.grid(alpha=0.25, axis='y')
        plt.show()
    """))

    cells.append(md("## 5. Rolling 4-fold mean Sharpe"))

    cells.append(md("""
        Smoothed view over time. The agent and in-universe baselines move
        almost identically through every regime — consistent with the agent
        learning a near-equal-weight policy.
    """))

    cells.append(code("""
        fig, ax = plt.subplots(figsize=(10, 5))
        for col, label, colour in [
            ('agent_sharpe',         'PPO agent',       '#1f77b4'),
            ('buy_and_hold_sharpe',  'buy_and_hold',    '#2ca02c'),
            ('equal_weight_sharpe',  'equal_weight',    '#ff7f0e'),
            ('sp500_sharpe',         'S&P 500 (^GSPC)', '#7f7f7f'),
        ]:
            s = (per_fold_mean(col)
                 .replace([np.inf, -np.inf], np.nan)
                 .rolling(4, min_periods=1).mean())
            ax.plot(dates.values, s.values, label=label, color=colour, lw=1.6)
        ax.axhline(0, color='black', lw=0.5, alpha=0.4)
        ax.set_title('Rolling 4-fold mean Sharpe')
        ax.set_ylabel('Mean Sharpe')
        ax.set_xlabel('Test-window start')
        ax.legend(loc='lower right', frameon=False)
        ax.grid(alpha=0.25)
        plt.show()
    """))

    cells.append(md("## 6. Rolling hit rate vs each baseline"))

    cells.append(md("""
        The most diagnostic plot. The 50% line is parity. Persistent
        deviations either way are interesting; the 2014-2022 stretch where
        the agent loses to in-universe baselines >60% of the time is the
        clearest signal in this CSV.
    """))

    cells.append(code("""
        fig, ax = plt.subplots(figsize=(10, 5))
        fold_idx = sorted(df['fold'].unique())
        for col, label, colour in [
            ('buy_and_hold_sharpe',  'agent > buy_and_hold',    '#2ca02c'),
            ('equal_weight_sharpe',  'agent > equal_weight',    '#ff7f0e'),
            ('sp500_sharpe',         'agent > S&P 500 (^GSPC)', '#7f7f7f'),
        ]:
            comp = df[['fold', 'agent_sharpe', col]].replace([np.inf, -np.inf], np.nan).dropna()
            wins = ((comp['agent_sharpe'] > comp[col])
                    .groupby(comp['fold']).mean()
                    .reindex(fold_idx))
            rolled = wins.rolling(8, min_periods=1).mean() * 100
            ax.plot(dates.values, rolled.values, label=label, color=colour, lw=1.6)
        ax.axhline(50, color='black', lw=0.5, alpha=0.4, linestyle='--')
        ax.set_title('Rolling 8-fold hit rate (50% line = parity)')
        ax.set_ylabel('Agent wins (%)')
        ax.set_xlabel('Test-window start')
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right', frameon=False)
        ax.grid(alpha=0.25)
        plt.show()
    """))

    cells.append(md("## 7. Seed dispersion — how stable is the agent?"))

    cells.append(md("""
        For each fold, what is the spread between the best and worst seed?
        Tight spread = stable training; wide spread = run-to-run lottery.
    """))

    cells.append(code("""
        per_fold = (df.groupby('fold')['agent_sharpe']
                      .agg(['min', 'mean', 'max', 'std']))
        per_fold['spread'] = per_fold['max'] - per_fold['min']

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(per_fold['spread'].dropna(), bins=30, color='#1f77b4', alpha=0.7)
        axes[0].set_title('Distribution of seed-spread per fold')
        axes[0].set_xlabel('max(Sharpe) − min(Sharpe) across seeds')
        axes[0].set_ylabel('Number of folds')
        axes[0].grid(alpha=0.25)

        axes[1].scatter(per_fold['mean'], per_fold['spread'], s=12, alpha=0.6, color='#1f77b4')
        axes[1].set_title('Spread vs mean Sharpe (per fold)')
        axes[1].set_xlabel('Mean Sharpe across seeds')
        axes[1].set_ylabel('Seed spread')
        axes[1].grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

        print(f'median seed-spread: {per_fold.spread.median():.3f}')
        print(f'mean   seed-spread: {per_fold.spread.mean():.3f}')
        print(f'>1.0 spread folds : {(per_fold.spread > 1.0).sum()} / {len(per_fold)}')
    """))

    cells.append(md("## 8. Per-fold detail (sortable)"))

    cells.append(md("""
        Each row is one fold's mean across seeds. Sort by `agent_minus_eq` to
        find the folds where the agent most beat / lost to the equal-weight
        baseline.
    """))

    cells.append(code("""
        per_fold_full = (df.groupby('fold')
                           .agg(test_start=('test_start', 'first'),
                                test_end=('test_end', 'first'),
                                agent_sharpe=('agent_sharpe', 'mean'),
                                bh_sharpe=('buy_and_hold_sharpe', 'mean'),
                                eq_sharpe=('equal_weight_sharpe', 'mean'),
                                sp500_sharpe=('sp500_sharpe', 'mean'),
                                agent_return=('agent_return', 'mean'))
                           .round(3))
        per_fold_full['agent_minus_eq'] = (
            per_fold_full['agent_sharpe'] - per_fold_full['eq_sharpe']
        ).round(3)
        per_fold_full.sort_values('agent_minus_eq', ascending=False).head(10)
    """))

    cells.append(code("""
        per_fold_full.sort_values('agent_minus_eq').head(10)
    """))

    cells.append(md("## 9. What this notebook does NOT do"))

    cells.append(md("""
        - **No statistical test** of whether the per-fold Sharpe deltas are
          distinguishable from zero. With 73 folds × 3 seeds, a paired
          bootstrap would be the natural next step. Not implemented here.
        - **No ablations** — same features, same reward, same architecture
          throughout. The result is a property of *one* config, not RL on
          this universe in general.
        - **No inspection of the learned policy.** The agent's per-step
          weights are not in this CSV. To peek at allocations, run the
          backtest interactively against `PortfolioEnv` and inspect
          `env.get_portfolio_history()`.

        See `README.md` → "Limitations & Honest Caveats" for the full list.
    """))

    nb.cells = cells
    return nb


def main() -> int:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    nbf.write(nb, NOTEBOOK_PATH)
    print(f'Wrote {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)} '
          f'({len(nb.cells)} cells)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
