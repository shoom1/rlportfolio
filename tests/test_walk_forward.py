"""Tests for evaluation.walk_forward (config validation, fold layout, summary)."""

import numpy as np
import pandas as pd
import pytest

from evaluation.walk_forward import WalkForwardConfig, WalkForwardEvaluator


# ---------------- WalkForwardConfig validation ----------------

class TestWalkForwardConfigValidation:

    def test_defaults_are_valid(self):
        wf = WalkForwardConfig()
        assert wf.t_min_days == 756
        assert wf.stride_days == 63
        assert wf.tau_days == 63

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
