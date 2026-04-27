"""Tests for evaluation.walk_forward (config validation, fold layout, summary)."""

import numpy as np
import pandas as pd
import pytest

import evaluation.walk_forward as walk_forward_module
from evaluation.walk_forward import WalkForwardConfig, WalkForwardEvaluator


# ---------------- WalkForwardConfig validation ----------------

class TestWalkForwardConfigValidation:

    def test_defaults_are_valid(self):
        wf = WalkForwardConfig()
        assert wf.t_min_days == 756
        assert wf.stride_days == 63
        assert wf.tau_days == 63
        assert wf.seed_values() == [42]

    @pytest.mark.parametrize("field,value", [
        ('t_min_days', 0),
        ('t_min_days', -1),
        ('stride_days', 0),
        ('tau_days', 0),
        ('selection_days', -1),
        ('n_workers', 0),
    ])
    def test_rejects_non_positive_knobs(self, field, value):
        kwargs = {field: value}
        with pytest.raises(ValueError, match=field):
            WalkForwardConfig(**kwargs)

    def test_rejects_selection_eating_all_of_train(self):
        """t_min_days must leave some training data outside the selection slice."""
        with pytest.raises(ValueError, match="selection_days"):
            WalkForwardConfig(t_min_days=100, selection_days=100)

    def test_rejects_unknown_baseline(self):
        with pytest.raises(ValueError, match="Unknown baselines"):
            WalkForwardConfig(baselines=['buy_and_hold', 'made_up_strategy'])

    def test_accepts_single_baseline(self):
        wf = WalkForwardConfig(baselines=['equal_weight'])
        assert wf.baselines == ['equal_weight']

    def test_accepts_explicit_seeds(self):
        wf = WalkForwardConfig(seeds=[7, 11, 13])
        assert wf.seed_values() == [7, 11, 13]

    def test_rejects_empty_seeds(self):
        with pytest.raises(ValueError, match="seeds"):
            WalkForwardConfig(seeds=[])

    def test_rejects_duplicate_seeds(self):
        with pytest.raises(ValueError, match="seeds"):
            WalkForwardConfig(seeds=[42, 42])


# ---------------- compute_folds (pure layout logic) ----------------

class TestComputeFolds:

    def test_basic_quarterly_layout(self):
        wf = WalkForwardConfig(
            t_min_days=100, stride_days=20, tau_days=20, selection_days=10,
        )
        folds = WalkForwardEvaluator.compute_folds(wf, n_dates=200)
        # Expected tk values: 100, 120, 140, 160, 180
        # Each needs tk + tau_days <= 200, so last valid tk is 180 (tests 180..200).
        assert len(folds) == 5
        for i, (idx, train_start, sel_start, test_start, test_end) in enumerate(folds):
            assert idx == i
            assert train_start == 0
            assert test_start == 100 + i * 20
            assert test_end == test_start + 20
            assert sel_start == test_start - 10

    def test_insufficient_data_returns_empty(self):
        wf = WalkForwardConfig(
            t_min_days=100, stride_days=20, tau_days=20, selection_days=10,
        )
        # Need t_min + tau = 120 trading days; 90 is not enough.
        assert WalkForwardEvaluator.compute_folds(wf, n_dates=90) == []

    def test_exactly_one_fold_when_data_just_fits(self):
        wf = WalkForwardConfig(
            t_min_days=100, stride_days=20, tau_days=20, selection_days=10,
        )
        folds = WalkForwardEvaluator.compute_folds(wf, n_dates=120)
        assert len(folds) == 1
        assert folds[0][3:5] == (100, 120)  # test_start, test_end

    def test_tau_larger_than_stride_creates_overlapping_tests(self):
        """With tau > stride, test windows overlap — still valid."""
        wf = WalkForwardConfig(
            t_min_days=100, stride_days=20, tau_days=40, selection_days=10,
        )
        folds = WalkForwardEvaluator.compute_folds(wf, n_dates=200)
        # tk progression: 100, 120, 140, 160 (last where tk+40 <= 200)
        assert len(folds) == 4
        # Verify overlap
        assert folds[0][4] > folds[1][3]  # fold 0 test_end > fold 1 test_start


# ---------------- summarise ----------------

class TestSummarise:

    def _toy_df(self):
        return pd.DataFrame({
            'fold': [0, 1, 2, 3],
            'agent_sharpe': [1.0, 2.0, 3.0, 4.0],
            'agent_return': [0.01, 0.02, 0.03, 0.04],
            'agent_dd': [-0.1, -0.05, -0.08, -0.02],
            'agent_vol': [0.15, 0.12, 0.18, 0.10],
            'buy_and_hold_sharpe': [0.5, 2.5, 2.5, 4.5],
            'buy_and_hold_return': [0.005, 0.025, 0.025, 0.045],
            'equal_weight_sharpe': [1.5, 1.5, 2.8, 3.5],
            'equal_weight_return': [0.015, 0.015, 0.028, 0.035],
        })

    def test_agent_stats(self):
        stats = WalkForwardEvaluator.summarise(
            self._toy_df(), baselines=['buy_and_hold', 'equal_weight'],
        )
        assert stats['n_folds'] == 4
        assert stats['n_runs'] == 4
        assert stats['n_successful_folds'] == 4
        assert stats['n_successful_runs'] == 4
        assert stats['n_failed_folds'] == 0
        assert stats['n_failed_runs'] == 0
        assert stats['agent_sharpe']['mean'] == pytest.approx(2.5)
        assert stats['agent_sharpe']['median'] == pytest.approx(2.5)
        assert stats['agent_sharpe']['min'] == 1.0
        assert stats['agent_sharpe']['max'] == 4.0

    def test_hit_rate_counts_strict_inequality(self):
        stats = WalkForwardEvaluator.summarise(
            self._toy_df(), baselines=['buy_and_hold'],
        )
        # agent_sharpe: [1.0, 2.0, 3.0, 4.0]
        # bh_sharpe:    [0.5, 2.5, 2.5, 4.5]
        # agent > bh:   [T,   F,   T,   F]   -> 2/4 = 50%
        assert stats['buy_and_hold_hit_rate'] == pytest.approx(0.5)

    def test_skips_baselines_not_in_df(self):
        stats = WalkForwardEvaluator.summarise(
            self._toy_df(), baselines=['buy_and_hold', 'momentum_20'],
        )
        assert 'buy_and_hold_sharpe' in stats
        assert 'momentum_20_sharpe' not in stats

    def test_inf_and_nan_are_stripped(self):
        df = pd.DataFrame({
            'fold': [0, 1, 2],
            'agent_sharpe': [1.0, np.inf, np.nan],
            'buy_and_hold_sharpe': [0.5, 1.0, 2.0],
        })
        stats = WalkForwardEvaluator.summarise(df, baselines=['buy_and_hold'])
        # inf + nan stripped -> only 1.0 remains
        assert stats['agent_sharpe']['mean'] == 1.0
        assert stats['agent_sharpe']['min'] == 1.0
        assert stats['agent_sharpe']['max'] == 1.0

    def test_multi_seed_stats_report_runs_and_fold_means(self):
        df = pd.DataFrame({
            'fold': [0, 0, 1, 1],
            'seed': [7, 11, 7, 11],
            'status': ['success', 'success', 'success', 'success'],
            'agent_sharpe': [1.0, 3.0, 2.0, 4.0],
            'buy_and_hold_sharpe': [0.5, 0.5, 3.0, 3.0],
        })

        stats = WalkForwardEvaluator.summarise(df, baselines=['buy_and_hold'])

        assert stats['n_folds'] == 2
        assert stats['n_runs'] == 4
        assert stats['n_successful_folds'] == 2
        assert stats['n_successful_runs'] == 4
        assert stats['agent_sharpe']['mean'] == pytest.approx(2.5)
        # Per-fold means are fold 0: 2.0, fold 1: 3.0.
        assert stats['agent_sharpe_by_fold']['mean'] == pytest.approx(2.5)
        assert stats['agent_sharpe_by_fold']['std'] == pytest.approx(np.sqrt(0.5))
        # agent > buy_and_hold by successful run: [T, T, F, T] -> 3/4.
        assert stats['buy_and_hold_hit_rate'] == pytest.approx(0.75)

    def test_failed_folds_are_excluded_from_performance_stats(self):
        df = pd.concat([
            self._toy_df().assign(
                status='success',
                error_type='',
                error_message='',
                error_traceback='',
            ),
            pd.DataFrame([{
                'fold': 4,
                'status': 'failed',
                'agent_sharpe': np.nan,
                'agent_return': np.nan,
                'agent_dd': np.nan,
                'agent_vol': np.nan,
                'buy_and_hold_sharpe': np.nan,
                'buy_and_hold_return': np.nan,
                'equal_weight_sharpe': np.nan,
                'equal_weight_return': np.nan,
                'error_type': 'RuntimeError',
                'error_message': 'training failed',
                'error_traceback': 'traceback',
            }]),
        ], ignore_index=True)

        stats = WalkForwardEvaluator.summarise(
            df, baselines=['buy_and_hold', 'equal_weight'],
        )

        assert stats['n_folds'] == 5
        assert stats['n_runs'] == 5
        assert stats['n_successful_folds'] == 4
        assert stats['n_successful_runs'] == 4
        assert stats['n_failed_folds'] == 1
        assert stats['n_failed_runs'] == 1
        assert stats['agent_sharpe']['mean'] == pytest.approx(2.5)
        assert stats['buy_and_hold_hit_rate'] == pytest.approx(0.5)


# ---------------- run failure recording ----------------

class TestRunFailureRecording:

    def test_run_records_failed_fold_and_continues(self, monkeypatch):
        all_dates = pd.date_range('2024-01-01', periods=8, freq='D')
        folds = [
            (0, 0, 2, 3, 4),
            (1, 0, 3, 4, 5),
        ]
        success_row = {
            'fold': 0,
            'seed': 42,
            'train_start': '2024-01-01',
            'sel_start': '2024-01-03',
            'test_start': '2024-01-04',
            'test_end': '2024-01-04',
            'status': 'success',
            'agent_sharpe': 1.25,
            'agent_return': 0.02,
            'agent_dd': -0.01,
            'agent_vol': 0.10,
            'buy_and_hold_sharpe': 0.75,
            'buy_and_hold_return': 0.01,
            'error_type': '',
            'error_message': '',
            'error_traceback': '',
        }

        class FakeFuture:
            def __init__(self, row=None, exc=None):
                self.row = row
                self.exc = exc

            def result(self):
                if self.exc is not None:
                    raise self.exc
                return self.row

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args):
                fold = args[:-1]
                if fold[0] == 0:
                    return FakeFuture(row=success_row)
                return FakeFuture(exc=RuntimeError('fold exploded'))

        evaluator = WalkForwardEvaluator(
            cfg=object(),
            wf=WalkForwardConfig(
                t_min_days=3,
                stride_days=1,
                tau_days=1,
                selection_days=1,
                n_workers=1,
                baselines=['buy_and_hold'],
            ),
            data_fetcher=object(),
            feature_engineer=object(),
        )
        monkeypatch.setattr(
            evaluator,
            'prepare_data',
            lambda: (pd.DataFrame(), [], all_dates),
        )
        monkeypatch.setattr(
            WalkForwardEvaluator,
            'compute_folds',
            staticmethod(lambda wf, n_dates: folds),
        )
        monkeypatch.setattr(walk_forward_module, 'ProcessPoolExecutor', FakeExecutor)
        monkeypatch.setattr(
            walk_forward_module,
            'as_completed',
            lambda futures: list(futures),
        )

        df = evaluator.run()

        assert list(df['fold']) == [0, 1]
        assert list(df['seed']) == [42, 42]
        assert list(df['status']) == ['success', 'failed']

        failed = df[df['status'] == 'failed'].iloc[0]
        assert failed['fold'] == 1
        assert failed['seed'] == 42
        assert failed['test_start'] == '2024-01-05'
        assert failed['test_end'] == '2024-01-05'
        assert failed['error_type'] == 'RuntimeError'
        assert failed['error_message'] == 'fold exploded'
        assert 'fold exploded' in failed['error_traceback']
        assert np.isnan(failed['agent_sharpe'])
        assert np.isnan(failed['buy_and_hold_sharpe'])

    def test_run_expands_each_fold_across_all_seeds(self, monkeypatch):
        all_dates = pd.date_range('2024-01-01', periods=8, freq='D')
        folds = [
            (0, 0, 2, 3, 4),
            (1, 0, 3, 4, 5),
        ]
        submitted = []

        class FakeFuture:
            def __init__(self, row):
                self.row = row

            def result(self):
                return self.row

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args):
                fold = args[:-1]
                seed = args[-1]
                submitted.append((fold[0], seed))
                return FakeFuture({
                    'fold': fold[0],
                    'seed': seed,
                    'train_start': '2024-01-01',
                    'sel_start': '2024-01-03',
                    'test_start': f'2024-01-0{fold[0] + 4}',
                    'test_end': f'2024-01-0{fold[0] + 4}',
                    'status': 'success',
                    'agent_sharpe': float(seed),
                    'agent_return': 0.02,
                    'agent_dd': -0.01,
                    'agent_vol': 0.10,
                    'buy_and_hold_sharpe': 0.75,
                    'buy_and_hold_return': 0.01,
                    'error_type': '',
                    'error_message': '',
                    'error_traceback': '',
                })

        evaluator = WalkForwardEvaluator(
            cfg=object(),
            wf=WalkForwardConfig(
                t_min_days=3,
                stride_days=1,
                tau_days=1,
                selection_days=1,
                n_workers=1,
                seeds=[7, 11],
                baselines=['buy_and_hold'],
            ),
            data_fetcher=object(),
            feature_engineer=object(),
        )
        monkeypatch.setattr(
            evaluator,
            'prepare_data',
            lambda: (pd.DataFrame(), [], all_dates),
        )
        monkeypatch.setattr(
            WalkForwardEvaluator,
            'compute_folds',
            staticmethod(lambda wf, n_dates: folds),
        )
        monkeypatch.setattr(walk_forward_module, 'ProcessPoolExecutor', FakeExecutor)
        monkeypatch.setattr(
            walk_forward_module,
            'as_completed',
            lambda futures: list(futures),
        )

        df = evaluator.run()

        assert submitted == [(0, 7), (0, 11), (1, 7), (1, 11)]
        assert list(df[['fold', 'seed']].itertuples(index=False, name=None)) == submitted
