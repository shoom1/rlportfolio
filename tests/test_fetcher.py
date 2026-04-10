"""
Unit tests for data/fetcher.py - DataFetcher using findata.DataClient.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from data.fetcher import DataFetcher


def _make_mock_wide_data(tickers, n_days=10):
    """Create mock wide-format data matching findata output."""
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    rows = []
    for date in dates:
        for symbol in tickers:
            rows.append((date, symbol))
    index = pd.MultiIndex.from_tuples(rows, names=['date', 'symbol'])
    n = len(rows)
    return pd.DataFrame({
        'open': np.random.rand(n) * 100,
        'high': np.random.rand(n) * 100,
        'low': np.random.rand(n) * 100,
        'close': np.random.rand(n) * 100,
        'volume': np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=index)


class TestDataFetcher:
    """Tests for DataFetcher class."""

    @patch('data.fetcher.DataClient')
    def test_fetch_data_returns_multiindex(self, MockClient):
        """Test that fetch_data returns (date, ticker) MultiIndex."""
        mock_client = MockClient.return_value
        mock_client.get_data.return_value = _make_mock_wide_data(['AAPL'])

        fetcher = DataFetcher()
        result = fetcher.fetch_data(['AAPL'], '2023-01-01', '2023-01-10')

        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ['date', 'ticker']

    @patch('data.fetcher.DataClient')
    def test_fetch_data_has_ohlcv_columns(self, MockClient):
        """Test that result has lowercase OHLCV columns."""
        mock_client = MockClient.return_value
        mock_client.get_data.return_value = _make_mock_wide_data(['AAPL'])

        fetcher = DataFetcher()
        result = fetcher.fetch_data(['AAPL'], '2023-01-01', '2023-01-10')

        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in result.columns

    @patch('data.fetcher.DataClient')
    def test_fetch_data_multiple_tickers(self, MockClient):
        """Test fetching data for multiple tickers."""
        tickers = ['AAPL', 'MSFT']
        mock_client = MockClient.return_value
        mock_client.get_data.return_value = _make_mock_wide_data(tickers)

        fetcher = DataFetcher()
        result = fetcher.fetch_data(tickers, '2023-01-01', '2023-01-10')

        result_tickers = result.index.get_level_values('ticker').unique()
        assert len(result_tickers) == 2
        for ticker in tickers:
            assert ticker in result_tickers

    @patch('data.fetcher.DataClient')
    def test_fetch_data_empty_raises_error(self, MockClient):
        """Test that empty data raises ValueError."""
        mock_client = MockClient.return_value
        mock_client.get_data.return_value = pd.DataFrame()

        fetcher = DataFetcher()
        with pytest.raises(ValueError, match="No data found"):
            fetcher.fetch_data(['INVALID'], '2023-01-01', '2023-01-10')

    @patch('data.fetcher.DataClient')
    def test_fetch_data_renames_symbol_to_ticker(self, MockClient):
        """Test that findata's 'symbol' index is renamed to 'ticker'."""
        mock_client = MockClient.return_value
        mock_data = _make_mock_wide_data(['AAPL'])
        assert mock_data.index.names == ['date', 'symbol']  # findata format

        mock_client.get_data.return_value = mock_data

        fetcher = DataFetcher()
        result = fetcher.fetch_data(['AAPL'], '2023-01-01', '2023-01-10')

        assert 'ticker' in result.index.names
        assert 'symbol' not in result.index.names

    @patch('data.fetcher.DataClient')
    def test_get_latest_data(self, MockClient):
        """Test get_latest_data method."""
        mock_client = MockClient.return_value
        mock_client.get_data.return_value = _make_mock_wide_data(['AAPL'])

        fetcher = DataFetcher()
        result = fetcher.get_latest_data(['AAPL'], days=365)

        assert isinstance(result, pd.DataFrame)
        mock_client.get_data.assert_called_once()

    @patch('data.fetcher.DataClient')
    def test_get_latest_data_passes_correct_dates(self, MockClient):
        """Test that get_latest_data calculates date range correctly."""
        mock_client = MockClient.return_value
        mock_client.get_data.return_value = _make_mock_wide_data(['AAPL'])

        fetcher = DataFetcher()
        fetcher.get_latest_data(['AAPL'], days=30)

        call_kwargs = mock_client.get_data.call_args
        assert call_kwargs.kwargs['start'] is not None
        assert call_kwargs.kwargs['end'] is not None
