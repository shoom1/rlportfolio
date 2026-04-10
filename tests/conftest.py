"""
Shared fixtures for pytest tests.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    data_frames = []
    np.random.seed(42)

    for ticker in tickers:
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        close_prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'open': close_prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'high': close_prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': close_prices * (1 + np.random.uniform(-0.02, 0, len(dates))),
            'close': close_prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        data_frames.append(df)

    combined = pd.concat(data_frames, ignore_index=True)
    combined = combined.set_index(['date', 'ticker'])

    return combined


@pytest.fixture
def sample_portfolio_history():
    """Create sample portfolio history for testing metrics."""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 252)
    values = 10000 * np.cumprod(1 + returns)

    history = pd.DataFrame({
        'date': dates,
        'value': values,
        'return': returns,
        'weights': [np.array([0.33, 0.33, 0.34]) for _ in range(252)],
        'cash': 1000.0,
        'transaction_cost': np.random.uniform(0, 10, 252)
    })

    return history


@pytest.fixture
def sample_tickers():
    """Sample ticker list."""
    return ['AAPL', 'MSFT', 'GOOGL']


@pytest.fixture
def feature_columns():
    """Sample feature column names."""
    return [
        'returns', 'log_returns', 'volatility_20',
        'close', 'sma_20', 'sma_50',
        'rsi_14', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'atr_14', 'volume_ratio'
    ]
