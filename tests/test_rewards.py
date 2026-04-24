"""
Unit tests for environment/rewards.py
"""

import pytest
import numpy as np
from environment.rewards import (
    SimpleReturnReward, LogReturnReward, SharpeReward, SortinoReward,
    RiskAdjustedReturnReward, DrawdownPenalizedReward, RewardFunctionFactory
)


class TestSimpleReturnReward:
    """Tests for SimpleReturnReward."""

    def test_compute_returns_portfolio_return(self):
        """Test that it returns the portfolio return."""
        reward_fn = SimpleReturnReward()
        reward = reward_fn.compute(
            portfolio_return=0.05,
            portfolio_value=10500,
            previous_value=10000
        )
        assert reward == 0.05

    def test_compute_with_negative_return(self):
        """Test with negative return."""
        reward_fn = SimpleReturnReward()
        reward = reward_fn.compute(
            portfolio_return=-0.03,
            portfolio_value=9700,
            previous_value=10000
        )
        assert reward == -0.03

    def test_reset_is_noop(self):
        """Test that stateless reward function has a callable reset()."""
        reward_fn = SimpleReturnReward()
        reward_fn.reset()  # Should not raise


class TestLogReturnReward:
    """Tests for LogReturnReward."""

    def test_compute_log_return(self):
        """Test log return computation."""
        reward_fn = LogReturnReward()
        reward = reward_fn.compute(
            portfolio_return=0.05,
            portfolio_value=10500,
            previous_value=10000
        )
        expected = np.log(10500 / 10000)
        assert np.isclose(reward, expected, atol=1e-6)

    def test_compute_with_zero_previous_value(self):
        """Test with zero previous value."""
        reward_fn = LogReturnReward()
        reward = reward_fn.compute(
            portfolio_return=0,
            portfolio_value=10000,
            previous_value=0
        )
        assert reward == 0.0

    def test_reset_is_noop(self):
        """Test that stateless reward function has a callable reset()."""
        reward_fn = LogReturnReward()
        reward_fn.reset()  # Should not raise


class TestSharpeReward:
    """Tests for SharpeReward."""

    def test_compute_builds_history(self):
        """Test that it builds return history."""
        reward_fn = SharpeReward(window=10)

        for i in range(5):
            reward_fn.compute(
                portfolio_return=0.01,
                portfolio_value=10000 * (1.01 ** (i + 1)),
                previous_value=10000 * (1.01 ** i)
            )

        assert len(reward_fn.returns_history) == 5

    def test_compute_maintains_window_size(self):
        """Test that history doesn't exceed window size."""
        reward_fn = SharpeReward(window=5)

        for i in range(10):
            reward_fn.compute(
                portfolio_return=0.01,
                portfolio_value=10000 * (1.01 ** (i + 1)),
                previous_value=10000 * (1.01 ** i)
            )

        assert len(reward_fn.returns_history) == 5

    def test_compute_with_insufficient_history(self):
        """Cold-start with <2 samples: fall back to the raw step return
        so the agent still gets a reward signal in the first step."""
        reward_fn = SharpeReward(window=10)
        reward = reward_fn.compute(
            portfolio_return=0.01,
            portfolio_value=10100,
            previous_value=10000
        )
        assert reward == 0.01

    def test_reset_clears_history(self):
        """Test that reset clears the history."""
        reward_fn = SharpeReward(window=10)

        for i in range(5):
            reward_fn.compute(
                portfolio_return=0.01,
                portfolio_value=10000 * (1.01 ** (i + 1)),
                previous_value=10000 * (1.01 ** i)
            )

        reward_fn.reset()
        assert len(reward_fn.returns_history) == 0


class TestSortinoReward:
    """Tests for SortinoReward."""

    def test_compute_builds_history(self):
        """Test that it builds return history."""
        reward_fn = SortinoReward(window=10)

        for i in range(5):
            reward_fn.compute(
                portfolio_return=0.01,
                portfolio_value=10000 * (1.01 ** (i + 1)),
                previous_value=10000 * (1.01 ** i)
            )

        assert len(reward_fn.returns_history) == 5

    def test_compute_with_no_downside(self):
        """Test with all positive returns."""
        reward_fn = SortinoReward(window=10)

        for i in range(10):
            reward_fn.compute(
                portfolio_return=0.01,
                portfolio_value=10000 * (1.01 ** (i + 1)),
                previous_value=10000 * (1.01 ** i)
            )

        reward = reward_fn.compute(
            portfolio_return=0.01,
            portfolio_value=11000,
            previous_value=10900
        )
        assert reward > 0

    def test_reset_clears_history(self):
        """Test that reset clears the history."""
        reward_fn = SortinoReward(window=10)

        for i in range(5):
            reward_fn.compute(
                portfolio_return=0.01,
                portfolio_value=10000 * (1.01 ** (i + 1)),
                previous_value=10000 * (1.01 ** i)
            )

        reward_fn.reset()
        assert len(reward_fn.returns_history) == 0


class TestRiskAdjustedReturnReward:
    """Tests for RiskAdjustedReturnReward."""

    def test_compute_penalizes_volatility(self):
        """Test that higher volatility leads to lower rewards."""
        reward_fn = RiskAdjustedReturnReward(risk_aversion=0.5, window=20)

        # Low volatility scenario
        for i in range(20):
            reward_fn.compute(
                portfolio_return=0.01,
                portfolio_value=10000 * (1.01 ** (i + 1)),
                previous_value=10000 * (1.01 ** i)
            )
        low_vol_reward = reward_fn.compute(
            portfolio_return=0.01,
            portfolio_value=11000,
            previous_value=10900
        )

        # High volatility scenario
        reward_fn.reset()
        np.random.seed(42)
        for i in range(20):
            ret = np.random.normal(0.01, 0.1)
            reward_fn.compute(
                portfolio_return=ret,
                portfolio_value=10000,
                previous_value=10000
            )
        high_vol_reward = reward_fn.compute(
            portfolio_return=0.01,
            portfolio_value=10100,
            previous_value=10000
        )

        assert low_vol_reward > high_vol_reward

    def test_reset_clears_history(self):
        """Test that reset clears the history."""
        reward_fn = RiskAdjustedReturnReward()

        for i in range(5):
            reward_fn.compute(
                portfolio_return=0.01,
                portfolio_value=10000 * (1.01 ** (i + 1)),
                previous_value=10000 * (1.01 ** i)
            )

        reward_fn.reset()
        assert len(reward_fn.returns_history) == 0


class TestDrawdownPenalizedReward:
    """Tests for DrawdownPenalizedReward."""

    def test_compute_tracks_peak(self):
        """Test that it tracks peak value."""
        reward_fn = DrawdownPenalizedReward(drawdown_penalty=2.0)

        reward_fn.compute(
            portfolio_return=0.1,
            portfolio_value=11000,
            previous_value=10000
        )

        assert reward_fn.peak_value == 11000

    def test_compute_penalizes_drawdown(self):
        """Test that drawdown is penalized."""
        reward_fn = DrawdownPenalizedReward(drawdown_penalty=2.0)

        # Go up
        reward1 = reward_fn.compute(
            portfolio_return=0.1,
            portfolio_value=11000,
            previous_value=10000
        )

        # Draw down
        reward2 = reward_fn.compute(
            portfolio_return=-0.05,
            portfolio_value=10450,
            previous_value=11000
        )

        assert reward2 < reward1

    def test_reset_clears_peak(self):
        """Test that reset clears peak value."""
        reward_fn = DrawdownPenalizedReward()

        reward_fn.compute(
            portfolio_return=0.1,
            portfolio_value=11000,
            previous_value=10000
        )

        reward_fn.reset()
        assert reward_fn.peak_value == 0.0


class TestRewardFunctionFactory:
    """Tests for RewardFunctionFactory."""

    def test_create_simple_return(self):
        """Test creating SimpleReturnReward."""
        reward_fn = RewardFunctionFactory.create('simple_return')
        assert isinstance(reward_fn, SimpleReturnReward)

    def test_create_log_return(self):
        """Test creating LogReturnReward."""
        reward_fn = RewardFunctionFactory.create('log_return')
        assert isinstance(reward_fn, LogReturnReward)

    def test_create_sharpe(self):
        """Test creating SharpeReward with parameters."""
        reward_fn = RewardFunctionFactory.create('sharpe', window=50)
        assert isinstance(reward_fn, SharpeReward)
        assert reward_fn.window == 50

    def test_create_sortino(self):
        """Test creating SortinoReward with parameters."""
        reward_fn = RewardFunctionFactory.create('sortino', window=40, risk_free_rate=0.03)
        assert isinstance(reward_fn, SortinoReward)
        assert reward_fn.window == 40

    def test_create_risk_adjusted(self):
        """Test creating RiskAdjustedReturnReward."""
        reward_fn = RewardFunctionFactory.create('risk_adjusted', risk_aversion=1.0)
        assert isinstance(reward_fn, RiskAdjustedReturnReward)
        assert reward_fn.risk_aversion == 1.0

    def test_create_drawdown_penalized(self):
        """Test creating DrawdownPenalizedReward."""
        reward_fn = RewardFunctionFactory.create('drawdown_penalized', drawdown_penalty=3.0)
        assert isinstance(reward_fn, DrawdownPenalizedReward)
        assert reward_fn.drawdown_penalty == 3.0

    def test_create_unknown_raises_error(self):
        """Test that unknown reward function raises error."""
        with pytest.raises(ValueError, match="Unknown reward function"):
            RewardFunctionFactory.create('unknown_reward')

    def test_list_available(self):
        """Test listing available reward functions."""
        available = RewardFunctionFactory.list_available()
        assert 'simple_return' in available
        assert 'log_return' in available
        assert 'sharpe' in available
        assert 'sortino' in available
        assert 'risk_adjusted' in available
        assert 'drawdown_penalized' in available

    def test_register_custom_reward(self):
        """Test registering custom reward function."""
        class CustomReward(SimpleReturnReward):
            pass

        RewardFunctionFactory.register('custom', CustomReward)
        reward_fn = RewardFunctionFactory.create('custom')
        assert isinstance(reward_fn, CustomReward)
