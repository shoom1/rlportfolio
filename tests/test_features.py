"""
Unit tests for data/features.py
"""

import pytest
import pandas as pd
import numpy as np
from data.features import (
    Feature, ReturnsFeature, VolatilityFeature, SMAFeature, EMAFeature,
    RSIFeature, MACDFeature, BollingerBandsFeature, ATRFeature, VolumeFeature,
    FeatureRegistry, FeatureEngineer
)


class TestReturnsFeature:
    """Tests for ReturnsFeature."""

    def test_compute_returns(self):
        """Test that returns are computed correctly."""
        df = pd.DataFrame({
            'Close': [100, 102, 101, 103, 105]
        })

        feature = ReturnsFeature()
        result = feature.compute(df)

        assert 'returns' in result.columns
        assert 'log_returns' in result.columns
        assert pd.isna(result['returns'].iloc[0])
        assert np.isclose(result['returns'].iloc[1], 0.02, atol=1e-6)

    def test_get_column_names(self):
        """Test column names."""
        feature = ReturnsFeature()
        columns = feature.get_column_names()
        assert columns == ['returns', 'log_returns']


class TestVolatilityFeature:
    """Tests for VolatilityFeature."""

    def test_compute_volatility(self):
        """Test volatility computation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        df = pd.DataFrame({
            'Close': 100 * np.cumprod(1 + returns)
        })

        feature = VolatilityFeature(window=20)
        result = feature.compute(df)

        assert 'volatility_20' in result.columns
        assert pd.isna(result['volatility_20'].iloc[:19]).all()
        assert not pd.isna(result['volatility_20'].iloc[20:]).any()

    def test_get_column_names(self):
        """Test column names."""
        feature = VolatilityFeature(window=30)
        columns = feature.get_column_names()
        assert columns == ['volatility_30']


class TestSMAFeature:
    """Tests for SMAFeature."""

    def test_compute_sma(self):
        """Test SMA computation."""
        df = pd.DataFrame({
            'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        })

        feature = SMAFeature(window=3)
        result = feature.compute(df)

        assert 'sma_3' in result.columns
        assert pd.isna(result['sma_3'].iloc[:2]).all()
        assert np.isclose(result['sma_3'].iloc[2], (100 + 102 + 101) / 3, atol=1e-6)

    def test_get_column_names(self):
        """Test column names."""
        feature = SMAFeature(window=50)
        columns = feature.get_column_names()
        assert columns == ['sma_50']


class TestEMAFeature:
    """Tests for EMAFeature."""

    def test_compute_ema(self):
        """Test EMA computation."""
        df = pd.DataFrame({
            'Close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        })

        feature = EMAFeature(span=3)
        result = feature.compute(df)

        assert 'ema_3' in result.columns
        assert not pd.isna(result['ema_3'].iloc[0])

    def test_get_column_names(self):
        """Test column names."""
        feature = EMAFeature(span=12)
        columns = feature.get_column_names()
        assert columns == ['ema_12']


class TestRSIFeature:
    """Tests for RSIFeature."""

    def test_compute_rsi(self):
        """Test RSI computation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        df = pd.DataFrame({
            'Close': 100 * np.cumprod(1 + returns)
        })

        feature = RSIFeature(length=14)
        result = feature.compute(df)

        assert 'rsi_14' in result.columns
        non_na_values = result['rsi_14'].dropna()
        assert (non_na_values >= 0).all()
        assert (non_na_values <= 100).all()

    def test_get_column_names(self):
        """Test column names."""
        feature = RSIFeature(length=14)
        columns = feature.get_column_names()
        assert columns == ['rsi_14']


class TestMACDFeature:
    """Tests for MACDFeature."""

    def test_compute_macd(self):
        """Test MACD computation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        df = pd.DataFrame({
            'Close': 100 * np.cumprod(1 + returns)
        })

        feature = MACDFeature(fast=12, slow=26, signal=9)
        result = feature.compute(df)

        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_hist' in result.columns

    def test_get_column_names(self):
        """Test column names."""
        feature = MACDFeature()
        columns = feature.get_column_names()
        assert columns == ['macd', 'macd_signal', 'macd_hist']


class TestBollingerBandsFeature:
    """Tests for BollingerBandsFeature."""

    def test_compute_bollinger_bands(self):
        """Test Bollinger Bands computation."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        df = pd.DataFrame({
            'Close': 100 * np.cumprod(1 + returns)
        })

        feature = BollingerBandsFeature(length=20, std=2.0)
        result = feature.compute(df)

        expected_cols = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent']
        for col in expected_cols:
            assert col in result.columns

        non_na_rows = result.dropna()
        assert (non_na_rows['bb_upper'] >= non_na_rows['bb_middle']).all()
        assert (non_na_rows['bb_middle'] >= non_na_rows['bb_lower']).all()

    def test_get_column_names(self):
        """Test column names."""
        feature = BollingerBandsFeature()
        columns = feature.get_column_names()
        assert columns == ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent']


class TestATRFeature:
    """Tests for ATRFeature."""

    def test_compute_atr(self):
        """Test ATR computation."""
        np.random.seed(42)
        df = pd.DataFrame({
            'High': 100 + np.random.rand(100) * 10,
            'Low': 100 - np.random.rand(100) * 10,
            'Close': 100 + np.random.randn(100) * 5
        })

        feature = ATRFeature(length=14)
        result = feature.compute(df)

        assert 'atr_14' in result.columns
        non_na_values = result['atr_14'].dropna()
        assert (non_na_values >= 0).all()

    def test_get_column_names(self):
        """Test column names."""
        feature = ATRFeature(length=14)
        columns = feature.get_column_names()
        assert columns == ['atr_14']


class TestVolumeFeature:
    """Tests for VolumeFeature."""

    def test_compute_volume_features(self):
        """Test volume-based features."""
        df = pd.DataFrame({
            'Volume': np.random.randint(1000000, 10000000, 100)
        })

        feature = VolumeFeature(window=20)
        result = feature.compute(df)

        assert 'volume_sma_20' in result.columns
        assert 'volume_ratio' in result.columns
        assert pd.isna(result['volume_sma_20'].iloc[:19]).all()

    def test_get_column_names(self):
        """Test column names."""
        feature = VolumeFeature(window=20)
        columns = feature.get_column_names()
        assert columns == ['volume_sma_20', 'volume_ratio']


class TestFeatureRegistry:
    """Tests for FeatureRegistry."""

    def test_register_and_get_feature(self):
        """Test registering and retrieving features."""
        registry = FeatureRegistry()
        feature = ReturnsFeature()

        registry.register(feature)
        retrieved = registry.get('returns')

        assert retrieved is feature

    def test_get_nonexistent_feature(self):
        """Test getting feature that doesn't exist."""
        registry = FeatureRegistry()
        assert registry.get('nonexistent') is None

    def test_list_features(self):
        """Test listing registered features."""
        registry = FeatureRegistry()
        registry.register(ReturnsFeature())
        registry.register(SMAFeature(20))

        features = registry.list_features()
        assert 'returns' in features
        assert 'sma_20' in features

    def test_compute_features(self):
        """Test computing multiple features."""
        registry = FeatureRegistry()
        registry.register(ReturnsFeature())
        registry.register(SMAFeature(3))

        df = pd.DataFrame({
            'Close': [100, 102, 101, 103, 105]
        })

        result = registry.compute_features(df, ['returns', 'sma_3'])

        assert 'returns' in result.columns
        assert 'log_returns' in result.columns
        assert 'sma_3' in result.columns

    def test_compute_unknown_feature_raises_error(self):
        """Test that computing unknown feature raises error."""
        registry = FeatureRegistry()
        df = pd.DataFrame({'Close': [100, 102, 101]})

        with pytest.raises(ValueError, match="Unknown feature"):
            registry.compute_features(df, ['unknown_feature'])


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""

    def test_init_registers_default_features(self):
        """Test that default features are registered."""
        engineer = FeatureEngineer()
        features = engineer.list_available_features()

        assert 'returns' in features
        assert 'volatility_20' in features
        assert 'sma_20' in features
        assert 'rsi_14' in features

    def test_add_custom_feature(self):
        """Test adding custom features."""
        custom_feature = SMAFeature(window=100)
        engineer = FeatureEngineer(custom_features=[custom_feature])

        features = engineer.list_available_features()
        assert 'sma_100' in features

    def test_compute_features_with_multiindex(self, sample_ohlcv_data):
        """Test computing features on MultiIndex data."""
        engineer = FeatureEngineer()
        feature_names = ['returns', 'sma_20']

        result = engineer.compute_features(sample_ohlcv_data, feature_names)

        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ['Date', 'Ticker']
        assert 'returns' in result.columns
        assert 'log_returns' in result.columns
        assert 'sma_20' in result.columns

    def test_get_default_feature_names(self):
        """Test getting default feature names."""
        engineer = FeatureEngineer()
        default_features = engineer.get_default_feature_names()

        assert isinstance(default_features, list)
        assert 'returns' in default_features
        assert 'volatility_20' in default_features

    def test_prepare_for_environment(self, sample_ohlcv_data):
        """Test preparing data for environment."""
        engineer = FeatureEngineer()
        features_data = engineer.compute_features(sample_ohlcv_data, ['returns', 'sma_20'])
        prepared = engineer.prepare_for_environment(features_data, lookback_window=20, normalize=True)

        assert isinstance(prepared, pd.DataFrame)
        assert not prepared.empty
        # Normalized columns will have NaN for first lookback_window periods
        assert prepared.dropna().isna().sum().sum() == 0

    def test_prepare_for_environment_without_normalization(self, sample_ohlcv_data):
        """Test preparing data without normalization."""
        engineer = FeatureEngineer()
        features_data = engineer.compute_features(sample_ohlcv_data, ['returns', 'sma_20'])
        prepared = engineer.prepare_for_environment(features_data, lookback_window=20, normalize=False)

        assert isinstance(prepared, pd.DataFrame)
        assert not prepared.empty

    def test_create_observation_columns(self):
        """Test creating observation columns list."""
        engineer = FeatureEngineer()

        normalized_cols = engineer.create_observation_columns(normalize=True)
        assert isinstance(normalized_cols, list)
        assert 'returns' in normalized_cols
        assert 'Close_norm' in normalized_cols

        unnormalized_cols = engineer.create_observation_columns(normalize=False)
        assert isinstance(unnormalized_cols, list)
        assert 'returns' in unnormalized_cols
        assert 'Close' in unnormalized_cols
