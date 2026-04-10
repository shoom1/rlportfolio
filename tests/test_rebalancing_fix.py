"""
Test to verify the rebalancing double-application bug fix.

This test explicitly checks that the cash ratio is applied correctly only once,
not twice as in the original buggy implementation.
"""

import pytest
import numpy as np
import pandas as pd
from environment.portfolio_env import PortfolioEnv
from environment.rewards import SimpleReturnReward
from data.features import FeatureEngineer


class TestRebalancingBugFix:
    """Tests verifying the rebalancing bug fix."""

    @pytest.fixture
    def env_data(self, sample_ohlcv_data):
        """Create environment-ready data."""
        engineer = FeatureEngineer()
        features_data = engineer.compute_features(sample_ohlcv_data, ['returns', 'sma_20'])
        prepared = engineer.prepare_for_environment(features_data, normalize=False)
        return prepared

    @pytest.fixture
    def simple_env(self, env_data, sample_tickers):
        """Create a simple environment for testing."""
        feature_cols = ['returns', 'close', 'sma_20']
        env = PortfolioEnv(
            data=env_data,
            feature_columns=feature_cols,
            tickers=sample_tickers,
            initial_balance=10000.0,
            transaction_cost=0.0,  # No costs to simplify test
            reward_function=SimpleReturnReward(),
            window_size=5
        )
        return env

    def test_rebalancing_cash_ratio_applied_once(self, simple_env):
        """
        Test that cash ratio is applied exactly once, not twice.

        This test verifies the fix for the bug where:
        - target_weights from softmax sum to (1 - cash_ratio)
        - Then we multiplied by target_asset_value = (1 - cash_ratio) * portfolio_value
        - Resulting in cash ratio being applied twice
        """
        obs, info = simple_env.reset()

        # Create action with explicit cash allocation
        # Action format: [asset1, asset2, asset3, cash]
        # Let's request 20% cash (logit = -1.0 will give ~20% after softmax)
        action = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float32)

        # Apply softmax to see what we expect
        exp_vals = np.exp(action - np.max(action))
        expected_normalized = exp_vals / np.sum(exp_vals)

        expected_cash_ratio = expected_normalized[-1]
        expected_asset_weights = expected_normalized[:-1]

        # Take a step
        obs, reward, terminated, truncated, info = simple_env.step(action)

        # Get current state
        portfolio_value = simple_env.portfolio_value
        cash = simple_env.cash
        asset_values = simple_env.holdings * simple_env._get_prices(simple_env.current_step)
        total_asset_value = np.sum(asset_values)

        # Verify cash ratio is correct (should be ~20%)
        actual_cash_ratio = cash / portfolio_value
        assert np.isclose(actual_cash_ratio, expected_cash_ratio, atol=0.01), \
            f"Cash ratio {actual_cash_ratio:.4f} doesn't match expected {expected_cash_ratio:.4f}"

        # Verify asset allocation is correct (should be ~80%)
        actual_asset_ratio = total_asset_value / portfolio_value
        expected_asset_ratio = 1 - expected_cash_ratio
        assert np.isclose(actual_asset_ratio, expected_asset_ratio, atol=0.01), \
            f"Asset ratio {actual_asset_ratio:.4f} doesn't match expected {expected_asset_ratio:.4f}"

        # Verify total allocation sums to 100%
        total_allocation = actual_cash_ratio + actual_asset_ratio
        assert np.isclose(total_allocation, 1.0, atol=0.01), \
            f"Total allocation {total_allocation:.4f} should be 1.0"

    def test_rebalancing_with_high_cash(self, simple_env):
        """Test rebalancing with high cash allocation (50%)."""
        obs, info = simple_env.reset()

        # Action requesting roughly 50% cash
        action = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # Apply softmax
        exp_vals = np.exp(action - np.max(action))
        expected_normalized = exp_vals / np.sum(exp_vals)
        expected_cash_ratio = expected_normalized[-1]

        # Take a step
        obs, reward, terminated, truncated, info = simple_env.step(action)

        # Verify allocations
        portfolio_value = simple_env.portfolio_value
        cash_ratio = simple_env.cash / portfolio_value
        asset_values = simple_env.holdings * simple_env._get_prices(simple_env.current_step)
        asset_ratio = np.sum(asset_values) / portfolio_value

        # With the fix, cash should be ~25% (1/4 from equal softmax weights)
        # Without the fix, it would be even higher due to double application
        assert np.isclose(cash_ratio, expected_cash_ratio, atol=0.02), \
            f"Cash ratio {cash_ratio:.4f} doesn't match expected {expected_cash_ratio:.4f}"

        assert np.isclose(cash_ratio + asset_ratio, 1.0, atol=0.01), \
            f"Total allocation {cash_ratio + asset_ratio:.4f} should be 1.0"

    def test_rebalancing_with_low_cash(self, simple_env):
        """Test rebalancing with low cash allocation (~5%)."""
        obs, info = simple_env.reset()

        # Action requesting roughly 5% cash (very negative weight for cash)
        action = np.array([1.0, 1.0, 1.0, -5.0], dtype=np.float32)

        # Apply softmax
        exp_vals = np.exp(action - np.max(action))
        expected_normalized = exp_vals / np.sum(exp_vals)
        expected_cash_ratio = expected_normalized[-1]

        # Take a step
        obs, reward, terminated, truncated, info = simple_env.step(action)

        # Verify allocations
        portfolio_value = simple_env.portfolio_value
        cash_ratio = simple_env.cash / portfolio_value
        asset_values = simple_env.holdings * simple_env._get_prices(simple_env.current_step)
        asset_ratio = np.sum(asset_values) / portfolio_value

        assert np.isclose(cash_ratio, expected_cash_ratio, atol=0.01), \
            f"Cash ratio {cash_ratio:.4f} doesn't match expected {expected_cash_ratio:.4f}"

        assert np.isclose(cash_ratio + asset_ratio, 1.0, atol=0.01), \
            f"Total allocation {cash_ratio + asset_ratio:.4f} should be 1.0"

        # Cash should be low (~1-5%)
        assert cash_ratio < 0.1, f"Cash ratio {cash_ratio:.4f} should be less than 10%"

    def test_asset_weights_correctly_distributed(self, simple_env):
        """Test that asset weights are correctly distributed among assets."""
        obs, info = simple_env.reset()

        # Action with unequal asset weights: favor asset 1, some asset 2, less asset 3
        # Plus 20% cash
        action = np.array([2.0, 1.0, 0.5, -1.0], dtype=np.float32)

        # Apply softmax to get expected weights
        exp_vals = np.exp(action - np.max(action))
        expected_normalized = exp_vals / np.sum(exp_vals)

        expected_cash_ratio = expected_normalized[-1]
        expected_asset_weights_raw = expected_normalized[:-1]

        # Normalize asset weights to sum to 1 (among assets only)
        expected_asset_weights = expected_asset_weights_raw / np.sum(expected_asset_weights_raw)

        # Take a step
        obs, reward, terminated, truncated, info = simple_env.step(action)

        # Get actual asset weights
        portfolio_value = simple_env.portfolio_value
        asset_values = simple_env.holdings * simple_env._get_prices(simple_env.current_step)
        total_asset_value = np.sum(asset_values)

        if total_asset_value > 0:
            actual_asset_weights = asset_values / total_asset_value

            # Verify each asset weight matches expected
            for i in range(len(expected_asset_weights)):
                assert np.isclose(actual_asset_weights[i], expected_asset_weights[i], atol=0.02), \
                    f"Asset {i} weight {actual_asset_weights[i]:.4f} doesn't match expected {expected_asset_weights[i]:.4f}"

    def test_multiple_rebalancing_steps_maintain_consistency(self, simple_env):
        """Test that multiple rebalancing steps maintain correct allocations."""
        obs, info = simple_env.reset()

        actions = [
            np.array([1.0, 1.0, 1.0, -2.0], dtype=np.float32),  # ~10% cash
            np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float32),  # ~20% cash
            np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),   # ~25% cash
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),   # ~25% cash (equal)
        ]

        for action in actions:
            if simple_env.current_step >= simple_env.n_steps - 1:
                break

            # Get expected from softmax
            exp_vals = np.exp(action - np.max(action))
            expected_normalized = exp_vals / np.sum(exp_vals)
            expected_cash_ratio = expected_normalized[-1]

            # Take step
            obs, reward, terminated, truncated, info = simple_env.step(action)

            # Verify allocation
            portfolio_value = simple_env.portfolio_value
            cash_ratio = simple_env.cash / portfolio_value
            asset_values = simple_env.holdings * simple_env._get_prices(simple_env.current_step)
            asset_ratio = np.sum(asset_values) / portfolio_value

            assert np.isclose(cash_ratio, expected_cash_ratio, atol=0.02), \
                f"Step {simple_env.current_step}: Cash ratio {cash_ratio:.4f} != expected {expected_cash_ratio:.4f}"

            assert np.isclose(cash_ratio + asset_ratio, 1.0, atol=0.01), \
                f"Step {simple_env.current_step}: Total allocation {cash_ratio + asset_ratio:.4f} != 1.0"

            if terminated or truncated:
                break
