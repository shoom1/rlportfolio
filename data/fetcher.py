"""
Data fetcher module for downloading and caching market data.
Supports yfinance and Polygon.io as data sources.
"""

import os
import pickle
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import pandas as pd
import yfinance as yf
from pathlib import Path


class DataFetcher:
    """Fetches and caches historical market data."""

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize data fetcher.

        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        source: str = "yfinance"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for given tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_cache: Whether to use cached data if available
            source: Data source ('yfinance' or 'polygon')

        Returns:
            DataFrame with MultiIndex (date, ticker) and columns (Open, High, Low, Close, Volume)
        """
        cache_key = self._get_cache_key(tickers, start_date, end_date, source)
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        # Try to load from cache
        if use_cache and cache_path.exists():
            print(f"Loading data from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Fetch new data
        print(f"Fetching data for {len(tickers)} tickers from {source}...")
        if source == "yfinance":
            data = self._fetch_yfinance(tickers, start_date, end_date)
        elif source == "polygon":
            data = self._fetch_polygon(tickers, start_date, end_date)
        else:
            raise ValueError(f"Unknown data source: {source}")

        # Cache the data
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data cached to: {cache_path}")

        return data

    def _fetch_yfinance(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch data using yfinance."""
        data_frames = []

        for ticker in tickers:
            print(f"  Downloading {ticker}...")
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False
            )

            if df.empty:
                print(f"  Warning: No data for {ticker}")
                continue

            # Handle MultiIndex columns (newer yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Rename columns to standard format (remove spaces if any)
            df.columns = [col.replace(' ', '') for col in df.columns]

            df['Ticker'] = ticker
            df = df.reset_index()
            data_frames.append(df)

        if not data_frames:
            raise ValueError("No data fetched for any ticker")

        # Combine all tickers
        combined = pd.concat(data_frames, ignore_index=True)

        # Set MultiIndex
        combined = combined.set_index(['Date', 'Ticker'])

        # Select and reorder columns
        columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        combined = combined[columns]

        return combined

    def _fetch_polygon(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch data using Polygon.io."""
        try:
            from polygon import RESTClient
        except ImportError:
            raise ImportError("polygon-api-client not installed. Install with: pip install polygon-api-client")

        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set")

        client = RESTClient(api_key)
        data_frames = []

        for ticker in tickers:
            print(f"  Downloading {ticker}...")
            aggs = []
            for a in client.list_aggs(
                ticker,
                1,
                "day",
                start_date,
                end_date,
                limit=50000
            ):
                aggs.append(a)

            if not aggs:
                print(f"  Warning: No data for {ticker}")
                continue

            df = pd.DataFrame([{
                'Date': pd.Timestamp(a.timestamp, unit='ms'),
                'Open': a.open,
                'High': a.high,
                'Low': a.low,
                'Close': a.close,
                'Volume': a.volume,
                'Ticker': ticker
            } for a in aggs])

            data_frames.append(df)

        if not data_frames:
            raise ValueError("No data fetched for any ticker")

        # Combine all tickers
        combined = pd.concat(data_frames, ignore_index=True)
        combined = combined.set_index(['Date', 'Ticker'])

        return combined

    def _get_cache_key(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        source: str
    ) -> str:
        """Generate a cache key from parameters."""
        tickers_str = "_".join(sorted(tickers))
        return f"{source}_{tickers_str}_{start_date}_{end_date}"

    def get_latest_data(
        self,
        tickers: List[str],
        days: int = 365,
        source: str = "yfinance"
    ) -> pd.DataFrame:
        """
        Fetch recent historical data.

        Args:
            tickers: List of ticker symbols
            days: Number of days of historical data
            source: Data source to use

        Returns:
            DataFrame with historical data
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        return self.fetch_data(tickers, start_date, end_date, source=source)


if __name__ == "__main__":
    # Example usage
    fetcher = DataFetcher()

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    data = fetcher.get_latest_data(tickers, days=730)

    print(f"\nData shape: {data.shape}")
    print(f"\nFirst few rows:\n{data.head()}")
    print(f"\nData info:")
    print(data.info())
