"""
Unit tests for data/fetcher.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from data.fetcher import DataFetcher


class TestDataFetcher:
    """Tests for DataFetcher class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def fetcher(self, temp_cache_dir):
        """Create a DataFetcher with temp cache directory."""
        return DataFetcher(cache_dir=temp_cache_dir)

    def test_init_creates_cache_directory(self, temp_cache_dir):
        """Test that initialization creates cache directory."""
        cache_path = Path(temp_cache_dir) / "new_cache"
        fetcher = DataFetcher(cache_dir=str(cache_path))
        assert cache_path.exists()
        assert cache_path.is_dir()

    def test_get_cache_key(self, fetcher):
        """Test cache key generation."""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        source = 'yfinance'

        cache_key = fetcher._get_cache_key(tickers, start_date, end_date, source)

        assert isinstance(cache_key, str)
        assert source in cache_key
        assert start_date in cache_key
        assert end_date in cache_key
        for ticker in tickers:
            assert ticker in cache_key

    def test_get_cache_key_consistent_order(self, fetcher):
        """Test that cache key is consistent regardless of ticker order."""
        tickers1 = ['AAPL', 'MSFT', 'GOOGL']
        tickers2 = ['GOOGL', 'AAPL', 'MSFT']
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        source = 'yfinance'

        key1 = fetcher._get_cache_key(tickers1, start_date, end_date, source)
        key2 = fetcher._get_cache_key(tickers2, start_date, end_date, source)

        assert key1 == key2

    @patch('data.fetcher.yf.download')
    def test_fetch_yfinance_single_ticker(self, mock_download, fetcher):
        """Test fetching data for a single ticker using yfinance."""
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.rand(10) * 100,
            'High': np.random.rand(10) * 100,
            'Low': np.random.rand(10) * 100,
            'Close': np.random.rand(10) * 100,
            'Volume': np.random.randint(1000000, 10000000, 10)
        }, index=dates)
        mock_data.index.name = 'Date'
        mock_download.return_value = mock_data

        result = fetcher._fetch_yfinance(['AAPL'], '2023-01-01', '2023-01-10')

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ['Date', 'Ticker']
        assert 'AAPL' in result.index.get_level_values('Ticker')
        assert all(col in result.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

    @patch('data.fetcher.yf.download')
    def test_fetch_yfinance_multiple_tickers(self, mock_download, fetcher):
        """Test fetching data for multiple tickers."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.rand(10) * 100,
            'High': np.random.rand(10) * 100,
            'Low': np.random.rand(10) * 100,
            'Close': np.random.rand(10) * 100,
            'Volume': np.random.randint(1000000, 10000000, 10)
        }, index=dates)
        mock_data.index.name = 'Date'
        mock_download.return_value = mock_data

        tickers = ['AAPL', 'MSFT']
        result = fetcher._fetch_yfinance(tickers, '2023-01-01', '2023-01-10')

        assert isinstance(result, pd.DataFrame)
        result_tickers = result.index.get_level_values('Ticker').unique()
        assert len(result_tickers) == len(tickers)
        for ticker in tickers:
            assert ticker in result_tickers

    @patch('data.fetcher.yf.download')
    def test_fetch_yfinance_empty_data_raises_error(self, mock_download, fetcher):
        """Test that empty data raises ValueError."""
        mock_download.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No data fetched for any ticker"):
            fetcher._fetch_yfinance(['INVALID'], '2023-01-01', '2023-01-10')

    @patch('data.fetcher.yf.download')
    def test_fetch_data_with_cache(self, mock_download, fetcher):
        """Test that cached data is used when available."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.rand(10) * 100,
            'High': np.random.rand(10) * 100,
            'Low': np.random.rand(10) * 100,
            'Close': np.random.rand(10) * 100,
            'Volume': np.random.randint(1000000, 10000000, 10)
        }, index=dates)
        mock_data.index.name = 'Date'
        mock_download.return_value = mock_data

        # First fetch - should call yfinance
        result1 = fetcher.fetch_data(['AAPL'], '2023-01-01', '2023-01-10', use_cache=True)
        assert mock_download.call_count == 1

        # Second fetch - should use cache
        result2 = fetcher.fetch_data(['AAPL'], '2023-01-01', '2023-01-10', use_cache=True)
        assert mock_download.call_count == 1  # Should not call again

        # Results should be equal
        pd.testing.assert_frame_equal(result1, result2)

    @patch('data.fetcher.yf.download')
    def test_fetch_data_without_cache(self, mock_download, fetcher):
        """Test fetching without using cache."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.rand(10) * 100,
            'High': np.random.rand(10) * 100,
            'Low': np.random.rand(10) * 100,
            'Close': np.random.rand(10) * 100,
            'Volume': np.random.randint(1000000, 10000000, 10)
        }, index=dates)
        mock_data.index.name = 'Date'
        mock_download.return_value = mock_data

        # Fetch twice without cache
        fetcher.fetch_data(['AAPL'], '2023-01-01', '2023-01-10', use_cache=False)
        fetcher.fetch_data(['AAPL'], '2023-01-01', '2023-01-10', use_cache=False)

        assert mock_download.call_count == 2

    def test_fetch_data_unknown_source_raises_error(self, fetcher):
        """Test that unknown data source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown data source"):
            fetcher.fetch_data(['AAPL'], '2023-01-01', '2023-01-10', source='unknown')

    @patch('data.fetcher.yf.download')
    def test_get_latest_data(self, mock_download, fetcher):
        """Test get_latest_data method."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.rand(10) * 100,
            'High': np.random.rand(10) * 100,
            'Low': np.random.rand(10) * 100,
            'Close': np.random.rand(10) * 100,
            'Volume': np.random.randint(1000000, 10000000, 10)
        }, index=dates)
        mock_data.index.name = 'Date'
        mock_download.return_value = mock_data

        result = fetcher.get_latest_data(['AAPL'], days=365)

        assert isinstance(result, pd.DataFrame)
        assert mock_download.called
