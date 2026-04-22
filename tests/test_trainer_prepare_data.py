"""Regression test for PortfolioTrainer.prepare_data entry point.

Guards against C1: passing unsupported kwargs to DataFetcher.get_latest_data.
Uses PortfolioTrainer's dependency-injection API (I5) rather than monkeypatching.
"""

import pandas as pd

from training.train import PortfolioTrainer


class SpyFetcher:
    """Fetcher whose get_latest_data signature mirrors DataFetcher's.

    If prepare_data ever passes an unsupported kwarg, the call fails here.
    """

    def __init__(self, data: pd.DataFrame):
        self._data = data
        self.calls = []

    def get_latest_data(self, tickers, days=365):
        self.calls.append({'tickers': list(tickers), 'days': days})
        return self._data


class IdentityEngineer:
    """Bypass feature engineering — we only care about the fetcher call."""

    def compute_features(self, data, feature_names=None):
        return data

    def prepare_for_environment(self, data, lookback_window=20, normalize=True):
        return data

    def create_observation_columns(self, normalize=True):
        return []


def test_prepare_data_calls_fetcher_with_supported_kwargs_only(sample_ohlcv_data):
    spy = SpyFetcher(sample_ohlcv_data)
    trainer = PortfolioTrainer(
        data_fetcher=spy,
        feature_engineer=IdentityEngineer(),
    )
    trainer.cfg.data.tickers = ['AAPL', 'MSFT', 'GOOGL']

    result = trainer.prepare_data(train=True)

    assert isinstance(result, pd.DataFrame)
    assert len(spy.calls) == 1
    assert spy.calls[0]['tickers'] == ['AAPL', 'MSFT', 'GOOGL']
    assert spy.calls[0]['days'] == trainer.cfg.data.train_days


def test_prepare_data_uses_val_days_when_train_false(sample_ohlcv_data):
    spy = SpyFetcher(sample_ohlcv_data)
    trainer = PortfolioTrainer(
        data_fetcher=spy,
        feature_engineer=IdentityEngineer(),
    )
    trainer.cfg.data.tickers = ['AAPL', 'MSFT', 'GOOGL']

    trainer.prepare_data(train=False)

    assert spy.calls[0]['days'] == trainer.cfg.data.val_days


def test_injected_config_takes_precedence_over_config_path(tmp_path, sample_ohlcv_data):
    """A config dict passed directly should bypass any YAML on disk."""
    injected = {
        'data': {
            'tickers': ['AAPL'],
            'train_days': 42,
            'val_days': 7,
        },
    }
    spy = SpyFetcher(sample_ohlcv_data)
    trainer = PortfolioTrainer(
        config=injected,
        data_fetcher=spy,
        feature_engineer=IdentityEngineer(),
    )
    trainer.prepare_data(train=True)

    assert spy.calls[0]['tickers'] == ['AAPL']
    assert spy.calls[0]['days'] == 42


def test_injected_cfg_takes_precedence_over_dict(sample_ohlcv_data):
    """A pre-built TrainingConfig should bypass dict/YAML loading entirely."""
    from training.config import TrainingConfig, DataConfig

    cfg = TrainingConfig(data=DataConfig(tickers=['MSFT'], train_days=100, val_days=10))
    spy = SpyFetcher(sample_ohlcv_data)
    trainer = PortfolioTrainer(
        cfg=cfg,
        data_fetcher=spy,
        feature_engineer=IdentityEngineer(),
    )
    trainer.prepare_data(train=True)

    assert spy.calls[0]['tickers'] == ['MSFT']
    assert spy.calls[0]['days'] == 100
