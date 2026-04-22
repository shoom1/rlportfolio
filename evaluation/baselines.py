"""
Baseline portfolio strategies for comparison.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict

from environment.constants import CASH_SOFTMAX_BIAS, LOG_EPSILON


class BaselineStrategy(ABC):
    """Base class for baseline portfolio strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_action(self, env, step: int, **kwargs) -> np.ndarray:
        """
        Get portfolio action for current step.

        Args:
            env: Portfolio environment
            step: Current step number
            **kwargs: Additional strategy-specific parameters

        Returns:
            Action array (will be passed through softmax in env)
        """
        pass

    def reset(self, seed: int = None):
        """Reset strategy state if needed.

        Args:
            seed: Optional seed for stochastic strategies. No-op for deterministic ones.
        """
        pass


class EqualWeightStrategy(BaselineStrategy):
    """Equal weight portfolio, rebalanced every step."""

    def __init__(self, cash_weight: float = CASH_SOFTMAX_BIAS):
        """
        Initialize equal weight strategy.

        Args:
            cash_weight: Raw weight for cash (before softmax).
                        Default -5.0 gives ~0% cash allocation.
        """
        super().__init__('equal_weight')
        self.cash_weight = cash_weight

    def get_action(self, env, step: int, **kwargs) -> np.ndarray:
        """Return equal weights for all assets + cash dimension."""
        # Equal weights for assets, specified weight for cash
        action = np.ones(env.n_assets + 1)
        action[-1] = self.cash_weight  # Cash dimension
        return action


class BuyAndHoldStrategy(BaselineStrategy):
    """Buy and hold strategy - equal weights at start, no rebalancing."""

    def __init__(self, cash_weight: float = CASH_SOFTMAX_BIAS):
        """
        Initialize buy and hold strategy.

        Args:
            cash_weight: Raw weight for cash (before softmax).
                        Default -5.0 gives ~0% cash allocation.
        """
        super().__init__('buy_and_hold')
        self.first_step = True
        self.cash_weight = cash_weight

    def get_action(self, env, step: int, **kwargs) -> np.ndarray:
        """Return equal weights on first step, then let weights drift."""
        if self.first_step:
            self.first_step = False
            action = np.ones(env.n_assets + 1)
            action[-1] = self.cash_weight  # Cash dimension
            return action
        else:
            # Return inverse softmax of current weights to maintain them
            asset_weights = np.log(env.weights + LOG_EPSILON)
            # Maintain cash ratio as well
            cash_ratio = env.cash / env.portfolio_value if env.portfolio_value > 0 else 0.0
            cash_log_weight = np.log(cash_ratio + LOG_EPSILON)
            return np.append(asset_weights, cash_log_weight)

    def reset(self, seed: int = None):
        """Reset first step flag."""
        self.first_step = True


class RandomStrategy(BaselineStrategy):
    """Random portfolio weights each step."""

    def __init__(self, seed: int = None):
        super().__init__('random')
        self._initial_seed = seed
        self.rng = np.random.RandomState(seed)

    def get_action(self, env, step: int, **kwargs) -> np.ndarray:
        """Return random weights including cash."""
        # Random weights for assets + cash
        return self.rng.randn(env.n_assets + 1)

    def reset(self, seed: int = None):
        """Re-seed the RNG so Monte Carlo simulations differ across runs."""
        effective_seed = seed if seed is not None else self._initial_seed
        self.rng = np.random.RandomState(effective_seed)


class MomentumStrategy(BaselineStrategy):
    """Momentum strategy - weight by recent returns."""

    def __init__(self, lookback: int = 20, cash_weight: float = CASH_SOFTMAX_BIAS):
        """
        Initialize momentum strategy.

        Args:
            lookback: Number of periods for momentum calculation
            cash_weight: Raw weight for cash (before softmax).
                        Default -5.0 gives ~0% cash allocation.
        """
        super().__init__(f'momentum_{lookback}')
        self.lookback = lookback
        self.cash_weight = cash_weight

    def get_action(self, env, step: int, **kwargs) -> np.ndarray:
        """Weight assets by momentum (past returns)."""
        if step < self.lookback:
            # Not enough history, use equal weight
            action = np.ones(env.n_assets + 1)
            action[-1] = self.cash_weight
            return action

        # Get past returns for each asset
        returns = []
        for i, ticker in enumerate(env.tickers):
            past_prices = []
            for j in range(self.lookback):
                prices = env._get_prices(step - self.lookback + j)
                past_prices.append(prices[i])

            # Calculate momentum as total return over lookback
            momentum = (past_prices[-1] / past_prices[0]) - 1.0
            returns.append(momentum)

        returns = np.array(returns)

        # Use returns as weights (positive momentum gets higher weight)
        # Shift to ensure all positive for softmax
        weights = returns - np.min(returns) + 1.0

        # Add cash dimension
        return np.append(weights, self.cash_weight)


class MinimumVarianceStrategy(BaselineStrategy):
    """Minimum variance portfolio based on historical covariance."""

    def __init__(self, lookback: int = 60, cash_weight: float = CASH_SOFTMAX_BIAS):
        """
        Initialize minimum variance strategy.

        Args:
            lookback: Number of periods for covariance calculation
            cash_weight: Raw weight for cash (before softmax).
                        Default -5.0 gives ~0% cash allocation.
        """
        super().__init__(f'min_variance_{lookback}')
        self.lookback = lookback
        self.cash_weight = cash_weight

    def get_action(self, env, step: int, **kwargs) -> np.ndarray:
        """Compute minimum variance weights."""
        if step < self.lookback:
            action = np.ones(env.n_assets + 1)
            action[-1] = self.cash_weight
            return action

        # Get historical returns
        returns_history = []
        for j in range(1, self.lookback):
            prices_prev = env._get_prices(step - j - 1)
            prices_curr = env._get_prices(step - j)
            returns = (prices_curr - prices_prev) / prices_prev
            returns_history.append(returns)

        returns_matrix = np.array(returns_history)

        # Calculate covariance matrix
        cov_matrix = np.cov(returns_matrix.T)

        # Minimum variance weights: inv(Σ) @ 1 / (1^T @ inv(Σ) @ 1)
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(env.n_assets)
            weights = inv_cov @ ones
            weights = weights / np.sum(weights)

            # Convert to action (log for softmax) and add cash dimension
            asset_action = np.log(weights + LOG_EPSILON)
            return np.append(asset_action, self.cash_weight)
        except np.linalg.LinAlgError:
            # If singular, fall back to equal weight
            action = np.ones(env.n_assets + 1)
            action[-1] = self.cash_weight
            return action


class InverseVolatilityStrategy(BaselineStrategy):
    """Inverse volatility weighting."""

    def __init__(self, lookback: int = 30, cash_weight: float = CASH_SOFTMAX_BIAS):
        """
        Initialize inverse volatility strategy.

        Args:
            lookback: Number of periods for volatility calculation
            cash_weight: Raw weight for cash (before softmax).
                        Default -5.0 gives ~0% cash allocation.
        """
        super().__init__(f'inverse_vol_{lookback}')
        self.lookback = lookback
        self.cash_weight = cash_weight

    def get_action(self, env, step: int, **kwargs) -> np.ndarray:
        """Weight inversely to volatility."""
        if step < self.lookback:
            action = np.ones(env.n_assets + 1)
            action[-1] = self.cash_weight
            return action

        # Get historical returns for each asset
        volatilities = []
        for i in range(env.n_assets):
            returns = []
            for j in range(1, self.lookback):
                prices_prev = env._get_prices(step - j - 1)
                prices_curr = env._get_prices(step - j)
                ret = (prices_curr[i] - prices_prev[i]) / prices_prev[i]
                returns.append(ret)

            vol = np.std(returns) if len(returns) > 1 else 1.0
            volatilities.append(vol)

        volatilities = np.array(volatilities)

        # Inverse volatility weights
        inv_vol = 1.0 / (volatilities + LOG_EPSILON)

        # Add cash dimension
        return np.append(inv_vol, self.cash_weight)


class BaselineStrategyRegistry:
    """Registry for baseline strategies."""

    def __init__(self):
        self._strategies: Dict[str, BaselineStrategy] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default baseline strategies."""
        self.register(EqualWeightStrategy())
        self.register(BuyAndHoldStrategy())
        self.register(RandomStrategy())
        self.register(MomentumStrategy(lookback=20))
        self.register(MinimumVarianceStrategy(lookback=60))
        self.register(InverseVolatilityStrategy(lookback=30))

    def register(self, strategy: BaselineStrategy):
        """Register a baseline strategy."""
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> BaselineStrategy:
        """Get a strategy by name."""
        if name not in self._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {self.list_strategies()}")
        return self._strategies[name]

    def list_strategies(self):
        """List all registered strategies."""
        return list(self._strategies.keys())
