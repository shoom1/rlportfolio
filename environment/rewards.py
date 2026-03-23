"""
Reward function implementations for portfolio optimization.
"""

import numpy as np
from typing import Optional
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def compute(
        self,
        portfolio_return: float,
        portfolio_value: float,
        previous_value: float,
        **kwargs
    ) -> float:
        """
        Compute reward for current step.

        Args:
            portfolio_return: Return for current step
            portfolio_value: Current portfolio value
            previous_value: Previous portfolio value
            **kwargs: Additional arguments specific to reward function

        Returns:
            Reward value
        """
        pass

    def reset(self):
        """Reset any internal state. Override in stateful subclasses."""
        pass


class SimpleReturnReward(RewardFunction):
    """Simple return-based reward."""

    def compute(
        self,
        portfolio_return: float,
        portfolio_value: float,
        previous_value: float,
        **kwargs
    ) -> float:
        """Return the portfolio return as reward."""
        return portfolio_return


class LogReturnReward(RewardFunction):
    """Log return-based reward."""

    def compute(
        self,
        portfolio_return: float,
        portfolio_value: float,
        previous_value: float,
        **kwargs
    ) -> float:
        """Return the log return as reward."""
        if previous_value > 0:
            return np.log(portfolio_value / previous_value)
        return 0.0


class SharpeReward(RewardFunction):
    """Sharpe ratio-based reward (rolling window)."""

    def __init__(self, window: int = 30, risk_free_rate: float = 0.0):
        """
        Initialize Sharpe reward.

        Args:
            window: Rolling window for computing Sharpe ratio
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
        """
        self.window = window
        self.daily_rf_rate = risk_free_rate / 252  # Convert to daily
        self.returns_history = []

    def compute(
        self,
        portfolio_return: float,
        portfolio_value: float,
        previous_value: float,
        **kwargs
    ) -> float:
        """Compute Sharpe ratio over rolling window."""
        self.returns_history.append(portfolio_return)

        # Keep only recent window
        if len(self.returns_history) > self.window:
            self.returns_history.pop(0)

        # Need sufficient history
        if len(self.returns_history) < 2:
            return 0.0

        returns_array = np.array(self.returns_history)
        excess_returns = returns_array - self.daily_rf_rate

        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)

        if std_return > 0:
            sharpe = mean_return / std_return
            # Scale to reasonable range
            return sharpe
        return 0.0

    def reset(self):
        """Reset the returns history."""
        self.returns_history = []


class SortinoReward(RewardFunction):
    """Sortino ratio-based reward (penalizes only downside volatility)."""

    def __init__(self, window: int = 30, risk_free_rate: float = 0.0):
        """
        Initialize Sortino reward.

        Args:
            window: Rolling window for computing Sortino ratio
            risk_free_rate: Annual risk-free rate
        """
        self.window = window
        self.daily_rf_rate = risk_free_rate / 252
        self.returns_history = []

    def compute(
        self,
        portfolio_return: float,
        portfolio_value: float,
        previous_value: float,
        **kwargs
    ) -> float:
        """Compute Sortino ratio over rolling window."""
        self.returns_history.append(portfolio_return)

        if len(self.returns_history) > self.window:
            self.returns_history.pop(0)

        if len(self.returns_history) < 2:
            return 0.0

        returns_array = np.array(self.returns_history)
        excess_returns = returns_array - self.daily_rf_rate

        mean_return = np.mean(excess_returns)

        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
        else:
            downside_std = 0.0

        if downside_std > 0:
            sortino = mean_return / downside_std
            return sortino
        elif mean_return > 0:
            # No downside but positive returns
            return mean_return * 10  # Large positive reward
        return 0.0

    def reset(self):
        """Reset the returns history."""
        self.returns_history = []


class RiskAdjustedReturnReward(RewardFunction):
    """Risk-adjusted return with volatility penalty."""

    def __init__(self, risk_aversion: float = 0.5, window: int = 20):
        """
        Initialize risk-adjusted reward.

        Args:
            risk_aversion: Coefficient for risk penalty (higher = more risk averse)
            window: Window for computing volatility
        """
        self.risk_aversion = risk_aversion
        self.window = window
        self.returns_history = []

    def compute(
        self,
        portfolio_return: float,
        portfolio_value: float,
        previous_value: float,
        **kwargs
    ) -> float:
        """Compute return minus risk penalty."""
        self.returns_history.append(portfolio_return)

        if len(self.returns_history) > self.window:
            self.returns_history.pop(0)

        # Reward = return - risk_aversion * variance
        mean_return = np.mean(self.returns_history)
        variance = np.var(self.returns_history) if len(self.returns_history) > 1 else 0.0

        reward = portfolio_return - self.risk_aversion * variance
        return reward

    def reset(self):
        """Reset the returns history."""
        self.returns_history = []


class DrawdownPenalizedReward(RewardFunction):
    """Return with drawdown penalty."""

    def __init__(self, drawdown_penalty: float = 2.0):
        """
        Initialize drawdown-penalized reward.

        Args:
            drawdown_penalty: Multiplier for drawdown penalty
        """
        self.drawdown_penalty = drawdown_penalty
        self.peak_value = 0.0

    def compute(
        self,
        portfolio_return: float,
        portfolio_value: float,
        previous_value: float,
        **kwargs
    ) -> float:
        """Compute return with drawdown penalty."""
        # Update peak
        self.peak_value = max(self.peak_value, portfolio_value)

        # Compute drawdown
        if self.peak_value > 0:
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
        else:
            drawdown = 0.0

        # Reward = return - penalty * drawdown
        reward = portfolio_return - self.drawdown_penalty * drawdown
        return reward

    def reset(self):
        """Reset the peak value."""
        self.peak_value = 0.0


class RewardFunctionFactory:
    """Factory for creating reward functions."""

    _registry = {
        'simple_return': SimpleReturnReward,
        'log_return': LogReturnReward,
        'sharpe': SharpeReward,
        'sortino': SortinoReward,
        'risk_adjusted': RiskAdjustedReturnReward,
        'drawdown_penalized': DrawdownPenalizedReward,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> RewardFunction:
        """
        Create a reward function by name.

        Args:
            name: Name of reward function
            **kwargs: Parameters for the reward function

        Returns:
            RewardFunction instance
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown reward function: {name}. "
                           f"Available: {list(cls._registry.keys())}")

        return cls._registry[name](**kwargs)

    @classmethod
    def register(cls, name: str, reward_class: type):
        """Register a custom reward function."""
        cls._registry[name] = reward_class

    @classmethod
    def list_available(cls):
        """List available reward functions."""
        return list(cls._registry.keys())
