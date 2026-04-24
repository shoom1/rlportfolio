"""
Feature engineering module for computing technical indicators and derived features.
Uses a registry pattern for modular and extensible feature computation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Callable, Dict
import pandas_ta as ta
from abc import ABC, abstractmethod


class Feature(ABC):
    """Base class for feature computation."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the feature and add columns to dataframe.

        Args:
            df: DataFrame with OHLCV data for a single ticker

        Returns:
            DataFrame with new feature columns added
        """
        pass

    @abstractmethod
    def get_column_names(self) -> List[str]:
        """Return list of column names this feature produces."""
        pass


class ReturnsFeature(Feature):
    """Simple and log returns."""

    def __init__(self):
        super().__init__('returns')

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        return df

    def get_column_names(self) -> List[str]:
        return ['returns', 'log_returns']


class VolatilityFeature(Feature):
    """Rolling volatility."""

    def __init__(self, window: int = 20):
        super().__init__(f'volatility_{window}')
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        df[f'volatility_{self.window}'] = df['returns'].rolling(window=self.window).std()
        return df

    def get_column_names(self) -> List[str]:
        return [f'volatility_{self.window}']


class SMAFeature(Feature):
    """Simple Moving Average."""

    def __init__(self, window: int):
        super().__init__(f'sma_{window}')
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f'sma_{self.window}'] = df['close'].rolling(window=self.window).mean()
        return df

    def get_column_names(self) -> List[str]:
        return [f'sma_{self.window}']


class EMAFeature(Feature):
    """Exponential Moving Average."""

    def __init__(self, span: int):
        super().__init__(f'ema_{span}')
        self.span = span

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f'ema_{self.span}'] = df['close'].ewm(span=self.span, adjust=False).mean()
        return df

    def get_column_names(self) -> List[str]:
        return [f'ema_{self.span}']


class RSIFeature(Feature):
    """Relative Strength Index."""

    def __init__(self, length: int = 14):
        super().__init__(f'rsi_{length}')
        self.length = length

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = ta.rsi(df['close'], length=self.length)
        df[f'rsi_{self.length}'] = rsi
        return df

    def get_column_names(self) -> List[str]:
        return [f'rsi_{self.length}']


class MACDFeature(Feature):
    """MACD indicator."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f'macd_{fast}_{slow}_{signal}')
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        macd = ta.macd(df['close'], fast=self.fast, slow=self.slow, signal=self.signal)
        if macd is not None:
            # Find actual column names (pandas-ta naming may vary)
            macd_col = [c for c in macd.columns if c.startswith('MACD_')][0]
            signal_col = [c for c in macd.columns if c.startswith('MACDs_')][0]
            hist_col = [c for c in macd.columns if c.startswith('MACDh_')][0]

            df['macd'] = macd[macd_col]
            df['macd_signal'] = macd[signal_col]
            df['macd_hist'] = macd[hist_col]
        return df

    def get_column_names(self) -> List[str]:
        return ['macd', 'macd_signal', 'macd_hist']


class BollingerBandsFeature(Feature):
    """Bollinger Bands."""

    def __init__(self, length: int = 20, std: float = 2.0):
        super().__init__(f'bbands_{length}_{std}')
        self.length = length
        self.std = std

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        bbands = ta.bbands(df['close'], length=self.length, std=self.std)
        if bbands is not None:
            # Handle both old and new pandas-ta column naming
            # Old: BBU_20_2.0, New: BBU_20_2 or BBU_20_2.00
            possible_suffixes = [
                f'_{self.length}_{self.std}',
                f'_{self.length}_{int(self.std)}',
                f'_{self.length}_{self.std:.1f}',
                f'_{self.length}_{self.std:.2f}'
            ]

            # Find the actual column names
            upper_col = None
            for suffix in possible_suffixes:
                col = f'BBU{suffix}'
                if col in bbands.columns:
                    upper_col = col
                    middle_col = f'BBM{suffix}'
                    lower_col = f'BBL{suffix}'
                    break

            if upper_col is None:
                # Fallback: find columns that start with BB
                upper_cols = [c for c in bbands.columns if c.startswith('BBU_')]
                if upper_cols:
                    upper_col = upper_cols[0]
                    middle_col = upper_col.replace('BBU_', 'BBM_')
                    lower_col = upper_col.replace('BBU_', 'BBL_')

            if upper_col:
                df['bb_upper'] = bbands[upper_col]
                df['bb_middle'] = bbands[middle_col]
                df['bb_lower'] = bbands[lower_col]
                # Guard against zero denominators on flat-price windows:
                # bb_middle can be 0 for zero-priced series, and
                # (bb_upper - bb_lower) collapses to 0 when volatility is
                # zero. Replace 0 with NaN so results are NaN rather than
                # inf — downstream env fillna handles the NaN cleanly.
                band_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
                middle = df['bb_middle'].replace(0, np.nan)
                df['bb_width'] = band_range / middle
                df['bb_percent'] = (df['close'] - df['bb_lower']) / band_range
        return df

    def get_column_names(self) -> List[str]:
        return ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent']


class ATRFeature(Feature):
    """Average True Range."""

    def __init__(self, length: int = 14):
        super().__init__(f'atr_{length}')
        self.length = length

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        atr = ta.atr(df['high'], df['low'], df['close'], length=self.length)
        df[f'atr_{self.length}'] = atr
        return df

    def get_column_names(self) -> List[str]:
        return [f'atr_{self.length}']


class VolumeFeature(Feature):
    """Volume-based indicators."""

    def __init__(self, window: int = 20):
        super().__init__(f'volume_{window}')
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f'volume_sma_{self.window}'] = df['volume'].rolling(window=self.window).mean()
        df['volume_ratio'] = df['volume'] / df[f'volume_sma_{self.window}']
        return df

    def get_column_names(self) -> List[str]:
        return [f'volume_sma_{self.window}', 'volume_ratio']


class FeatureRegistry:
    """Registry for managing feature computation."""

    def __init__(self):
        self._features: Dict[str, Feature] = {}

    def register(self, feature: Feature) -> None:
        """Register a feature."""
        self._features[feature.name] = feature

    def get(self, name: str) -> Optional[Feature]:
        """Get a feature by name."""
        return self._features.get(name)

    def list_features(self) -> List[str]:
        """List all registered feature names."""
        return list(self._features.keys())

    def compute_features(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """
        Compute selected features.

        Args:
            df: DataFrame with OHLCV data for a single ticker
            feature_names: List of feature names to compute

        Returns:
            DataFrame with computed features
        """
        result = df.copy()
        for name in feature_names:
            feature = self.get(name)
            if feature is None:
                raise ValueError(f"Unknown feature: {name}")
            result = feature.compute(result)
        return result


class FeatureEngineer:
    """Computes technical indicators and features for market data."""

    def __init__(self, custom_features: Optional[List[Feature]] = None):
        """
        Initialize feature engineer with default and custom features.

        Args:
            custom_features: Optional list of custom Feature instances to register
        """
        self.registry = FeatureRegistry()
        self._register_default_features()

        if custom_features:
            for feature in custom_features:
                self.registry.register(feature)

    def _register_default_features(self):
        """Register default feature set."""
        # Returns and volatility
        self.registry.register(ReturnsFeature())
        self.registry.register(VolatilityFeature(window=20))

        # Moving averages
        self.registry.register(SMAFeature(window=20))
        self.registry.register(SMAFeature(window=50))
        self.registry.register(EMAFeature(span=12))
        self.registry.register(EMAFeature(span=26))

        # Momentum indicators
        self.registry.register(RSIFeature(length=14))
        self.registry.register(MACDFeature(fast=12, slow=26, signal=9))

        # Volatility indicators
        self.registry.register(BollingerBandsFeature(length=20, std=2.0))
        self.registry.register(ATRFeature(length=14))

        # Volume
        self.registry.register(VolumeFeature(window=20))

    def add_feature(self, feature: Feature) -> None:
        """Add a custom feature to the registry."""
        self.registry.register(feature)

    def list_available_features(self) -> List[str]:
        """List all available feature names."""
        return self.registry.list_features()

    def compute_features(
        self,
        data: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute technical indicators for market data.

        Args:
            data: DataFrame with MultiIndex (date, ticker) and OHLCV columns
            feature_names: List of feature names to compute. If None, computes all default features.

        Returns:
            DataFrame with original data plus computed features
        """
        if feature_names is None:
            feature_names = self.get_default_feature_names()

        # Get list of tickers
        tickers = data.index.get_level_values('ticker').unique()

        # Process each ticker separately
        processed_dfs = []
        for ticker in tickers:
            ticker_df = data.xs(ticker, level='ticker').copy()
            ticker_df = self.registry.compute_features(ticker_df, feature_names)
            ticker_df['ticker'] = ticker
            processed_dfs.append(ticker_df)

        # Combine back
        result = pd.concat(processed_dfs)
        result = result.reset_index().set_index(['date', 'ticker']).sort_index()

        return result

    def get_default_feature_names(self) -> List[str]:
        """Get default feature set for standard use."""
        return [
            'returns',
            'volatility_20',
            'sma_20',
            'sma_50',
            'ema_12',
            'ema_26',
            'rsi_14',
            'macd_12_26_9',
            'bbands_20_2.0',
            'atr_14',
            'volume_20'
        ]

    def prepare_for_environment(
        self,
        data: pd.DataFrame,
        lookback_window: int = 20,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Prepare features for the RL environment.

        Args:
            data: DataFrame with features
            lookback_window: Number of periods to keep for sequence models
            normalize: Whether to normalize features

        Returns:
            Processed DataFrame ready for environment
        """
        df = data.copy()

        # Per-ticker forward fill: fills intra-series gaps (holidays, missed
        # bars) WITHOUT cross-contaminating values between tickers. A plain
        # df.ffill() on a (date, ticker)-sorted index would carry one
        # ticker's value into another's NaN cell.
        df = df.groupby(level='ticker', group_keys=False).ffill()

        # Intentionally NO dropna here. Indicator warm-up (e.g. SMA_50
        # needs 50 bars) and normalization warm-up (pct_change over
        # lookback_window) leave leading NaN rows per ticker. Dropping
        # them globally truncates the entire multi-ticker frame to the
        # slowest warm-up — wasting data. Instead we keep all rows and
        # rely on PortfolioEnv._precompute_arrays, which fillna(0.0) on
        # feature columns and per-ticker ffills on close price. Warm-up
        # observations are noisy but their prices are valid.

        if normalize:
            # Normalize price-based features using percent change from lookback
            price_cols = ['open', 'high', 'low', 'close', 'sma_20', 'sma_50',
                         'ema_12', 'ema_26', 'bb_upper', 'bb_middle', 'bb_lower']

            for col in price_cols:
                if col in df.columns:
                    df[f'{col}_norm'] = df.groupby(level='ticker')[col].pct_change(lookback_window)

            # RSI is already 0-100, normalize to 0-1
            if 'rsi_14' in df.columns:
                df['rsi_14_norm'] = df['rsi_14'] / 100.0

            # ATR normalized by price
            if 'atr_14' in df.columns:
                df['atr_14_norm'] = df['atr_14'] / df['close']

        return df

    def create_observation_columns(self, normalize: bool = True) -> List[str]:
        """
        Get list of column names to use for environment observations.

        Args:
            normalize: Whether using normalized features

        Returns:
            List of column names
        """
        if normalize:
            return [
                'returns', 'log_returns', 'volatility_20',
                'close_norm', 'sma_20_norm', 'sma_50_norm',
                'rsi_14_norm', 'macd', 'macd_signal',
                'bb_width', 'bb_percent', 'atr_14_norm', 'volume_ratio'
            ]
        else:
            return [
                'returns', 'log_returns', 'volatility_20',
                'close', 'sma_20', 'sma_50',
                'rsi_14', 'macd', 'macd_signal',
                'bb_upper', 'bb_lower', 'atr_14', 'volume_ratio'
            ]


if __name__ == "__main__":
    # Example usage
    from .fetcher import DataFetcher

    fetcher = DataFetcher()
    engineer = FeatureEngineer()

    # Fetch data
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = fetcher.get_latest_data(tickers, days=365)

    # List available features
    print("Available features:")
    for feature_name in engineer.list_available_features():
        print(f"  - {feature_name}")

    # Compute features
    print("\nComputing features...")
    features_data = engineer.compute_features(data)

    print(f"\nFeatures shape: {features_data.shape}")
    print(f"\nColumns: {features_data.columns.tolist()}")
    print(f"\nSample data:\n{features_data.head(10)}")

    # Prepare for environment
    print("\nPreparing for environment...")
    env_data = engineer.prepare_for_environment(features_data)

    print(f"\nEnvironment-ready data shape: {env_data.shape}")
    print(f"\nMissing values: {env_data.isna().sum().sum()}")
