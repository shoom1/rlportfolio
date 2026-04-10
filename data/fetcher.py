"""
Data fetcher module - thin adapter around findata.DataClient.
Provides market data with (date, ticker) MultiIndex and lowercase OHLCV columns.
"""

from typing import List, Optional
import pandas as pd
from findata import DataClient


class DataFetcher:
    """Fetches historical market data via findata."""

    def __init__(self):
        self.client = DataClient()

    def fetch_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for given tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with MultiIndex (date, ticker) and columns
            (open, high, low, close, volume)
        """
        df = self.client.get_data(
            symbols=tickers,
            start=start_date,
            end=end_date,
            columns=['open', 'high', 'low', 'close', 'volume'],
            format='wide',
        )

        if df.empty:
            raise ValueError(f"No data found for {tickers} in [{start_date}, {end_date}]")

        # Rename index level: symbol → ticker
        df.index = df.index.rename({'symbol': 'ticker'})

        return df

    def get_latest_data(
        self,
        tickers: List[str],
        days: int = 365,
    ) -> pd.DataFrame:
        """
        Fetch recent historical data.

        Args:
            tickers: List of ticker symbols
            days: Number of days of historical data

        Returns:
            DataFrame with historical data
        """
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')

        return self.fetch_data(tickers, start_date, end_date)


if __name__ == "__main__":
    fetcher = DataFetcher()

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    data = fetcher.get_latest_data(tickers, days=730)

    print(f"\nData shape: {data.shape}")
    print(f"\nFirst few rows:\n{data.head()}")
    print(f"\nData info:")
    print(data.info())
