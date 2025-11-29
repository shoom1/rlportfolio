"""
Visualization functions for portfolio analysis.
Standalone functions that can work with BacktestResult objects or raw DataFrames.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union


def plot_strategy_comparison(
    results: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None
):
    """
    Plot comparison of multiple strategies.

    Args:
        results: Dictionary mapping strategy names to portfolio history DataFrames
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Portfolio Strategy Comparison', fontsize=16, fontweight='bold')

    # 1. Portfolio value over time
    ax = axes[0, 0]
    for name, history in results.items():
        ax.plot(history['date'], history['value'], label=name, linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title('Portfolio Value Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    for name, history in results.items():
        values = history['value'].values
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        ax.plot(history['date'], drawdown * 100, label=name, linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Returns distribution
    ax = axes[1, 0]
    returns_data = []
    labels = []
    for name, history in results.items():
        if 'return' in history.columns:
            returns_data.append(history['return'].values * 100)
            labels.append(name)
    if returns_data:
        ax.hist(returns_data, bins=50, alpha=0.6, label=labels)
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Returns Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. Cumulative returns
    ax = axes[1, 1]
    for name, history in results.items():
        cumulative_return = (history['value'] / history['value'].iloc[0] - 1) * 100
        ax.plot(history['date'], cumulative_return, label=name, linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('Cumulative Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def plot_portfolio_weights(
    history: pd.DataFrame,
    tickers: List[str],
    save_path: Optional[str] = None
):
    """
    Plot portfolio weights evolution over time.

    Args:
        history: Portfolio history DataFrame with 'weights' column
        tickers: List of ticker names
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract weights
    weights = np.array([w for w in history['weights']])

    # Create stacked area plot
    ax.stackplot(
        history['date'],
        weights.T,
        labels=tickers,
        alpha=0.8
    )

    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Weight')
    ax.set_title('Portfolio Allocation Over Time')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nWeights plot saved to: {save_path}")

    plt.show()


def plot_risk_return_scatter(
    all_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
):
    """
    Plot risk-return scatter plot for multiple strategies.

    Args:
        all_metrics: Dictionary of metrics for each strategy
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    strategies = list(all_metrics.keys())
    returns = [all_metrics[s]['annualized_return'] * 100 for s in strategies]
    volatilities = [all_metrics[s]['volatility'] * 100 for s in strategies]
    sharpe_ratios = [all_metrics[s]['sharpe_ratio'] for s in strategies]

    # Color by Sharpe ratio
    scatter = ax.scatter(
        volatilities,
        returns,
        c=sharpe_ratios,
        s=200,
        alpha=0.6,
        cmap='RdYlGn',
        edgecolors='black',
        linewidth=2
    )

    # Add labels
    for i, strategy in enumerate(strategies):
        ax.annotate(
            strategy,
            (volatilities[i], returns[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel('Volatility (%)', fontsize=12)
    ax.set_ylabel('Annualized Return (%)', fontsize=12)
    ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nRisk-return plot saved to: {save_path}")

    plt.show()


def plot_rolling_metrics(
    history: pd.DataFrame,
    window: int = 30,
    save_path: Optional[str] = None
):
    """
    Plot rolling performance metrics.

    Args:
        history: Portfolio history DataFrame with 'return' column
        window: Rolling window size
        save_path: Optional path to save plot
    """
    if 'return' not in history.columns:
        raise ValueError("History must contain 'return' column")

    returns = history['return'].values
    dates = history['date'].values

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Rolling Performance Metrics (Window={window})', fontsize=16, fontweight='bold')

    # 1. Rolling mean return
    ax = axes[0, 0]
    rolling_mean = pd.Series(returns).rolling(window=window).mean()
    ax.plot(dates, rolling_mean * 100, linewidth=2, color='blue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Return (%)')
    ax.set_title('Rolling Mean Return')
    ax.grid(True, alpha=0.3)

    # 2. Rolling volatility
    ax = axes[0, 1]
    rolling_std = pd.Series(returns).rolling(window=window).std() * np.sqrt(252) * 100
    ax.plot(dates, rolling_std, linewidth=2, color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (%)')
    ax.set_title('Rolling Volatility (Annualized)')
    ax.grid(True, alpha=0.3)

    # 3. Rolling Sharpe ratio
    ax = axes[1, 0]
    rolling_sharpe = (
        pd.Series(returns).rolling(window=window).mean() /
        pd.Series(returns).rolling(window=window).std() *
        np.sqrt(252)
    )
    ax.plot(dates, rolling_sharpe, linewidth=2, color='green')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Rolling Sharpe Ratio')
    ax.grid(True, alpha=0.3)

    # 4. Rolling win rate
    ax = axes[1, 1]
    rolling_win_rate = (
        pd.Series(returns > 0).rolling(window=window).mean() * 100
    )
    ax.plot(dates, rolling_win_rate, linewidth=2, color='purple')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Rolling Win Rate')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nRolling metrics plot saved to: {save_path}")

    plt.show()


def plot_monthly_returns_heatmap(
    history: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot monthly returns heatmap.

    Args:
        history: Portfolio history DataFrame with 'date' and 'return' columns
        save_path: Optional path to save plot
    """
    if 'return' not in history.columns or 'date' not in history.columns:
        raise ValueError("History must contain 'date' and 'return' columns")

    # Create DataFrame with date index
    df = history.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Calculate monthly returns
    monthly_returns = df['return'].resample('M').apply(lambda x: (1 + x).prod() - 1)

    # Create pivot table for heatmap
    monthly_returns_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values * 100
    })

    pivot = monthly_returns_df.pivot(index='month', columns='year', values='return')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Return (%)'},
        ax=ax
    )

    ax.set_xlabel('Year')
    ax.set_ylabel('Month')
    ax.set_title('Monthly Returns Heatmap (%)')
    ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nMonthly returns heatmap saved to: {save_path}")

    plt.show()


def plot_underwater(
    history: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot underwater (drawdown) chart.

    Args:
        history: Portfolio history DataFrame with 'date' and 'value' columns
        save_path: Optional path to save plot
    """
    if 'value' not in history.columns or 'date' not in history.columns:
        raise ValueError("History must contain 'date' and 'value' columns")

    values = history['value'].values
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(history['date'], drawdown, 0, alpha=0.3, color='red')
    ax.plot(history['date'], drawdown, linewidth=2, color='darkred')

    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Underwater Plot (Drawdown from Peak)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nUnderwater plot saved to: {save_path}")

    plt.show()
