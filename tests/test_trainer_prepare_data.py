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


def test_prepare_data_fetches_train_plus_val_once(sample_ohlcv_data):
    """prepare_data must fetch a single combined window of
    train_days + val_days so the two slices can be disjoint, and must not
    re-fetch on a second call."""
    spy = SpyFetcher(sample_ohlcv_data)
    trainer = PortfolioTrainer(
        data_fetcher=spy,
        feature_engineer=IdentityEngineer(),
    )
    trainer.cfg.data.tickers = ['AAPL', 'MSFT', 'GOOGL']

    train = trainer.prepare_data(train=True)
    val = trainer.prepare_data(train=False)

    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert len(spy.calls) == 1, "fetcher should only be called once (cached)"
    assert spy.calls[0]['tickers'] == ['AAPL', 'MSFT', 'GOOGL']
    expected_days = trainer.cfg.data.train_days + trainer.cfg.data.val_days
    assert spy.calls[0]['days'] == expected_days


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
    assert spy.calls[0]['days'] == 42 + 7  # train_days + val_days


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
    assert spy.calls[0]['days'] == 100 + 10  # train_days + val_days


def test_prepare_data_returns_disjoint_train_val(sample_ohlcv_data):
    """Train and validation date ranges must not overlap, and val must come
    after train. Previously both calls fetched recent data independently, so
    val was the tail of train — producing in-sample 'val' metrics."""
    from training.config import TrainingConfig, DataConfig

    # Fixture has 100 trading days (2023-01-01 to 2023-04-10). Pick sizes
    # that both fit inside it.
    cfg = TrainingConfig(
        data=DataConfig(
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            train_days=70,
            val_days=20,
        ),
    )
    spy = SpyFetcher(sample_ohlcv_data)
    trainer = PortfolioTrainer(
        cfg=cfg,
        data_fetcher=spy,
        feature_engineer=IdentityEngineer(),
    )

    train = trainer.prepare_data(train=True)
    val = trainer.prepare_data(train=False)

    train_dates = set(train.index.get_level_values('date').unique())
    val_dates = set(val.index.get_level_values('date').unique())

    assert train_dates, "train split should not be empty"
    assert val_dates, "val split should not be empty"
    assert not (train_dates & val_dates), "train and val dates must be disjoint"
    assert max(train_dates) < min(val_dates), "val must come strictly after train"
