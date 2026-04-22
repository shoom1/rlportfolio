"""Shared numeric constants used across the RL portfolio pipeline."""


DEFAULT_INITIAL_BALANCE: float = 10_000.0
"""Starting portfolio value in dollars for the default environment."""

DEFAULT_TRANSACTION_COST: float = 0.001
"""Proportional transaction cost (0.1%) applied when no custom cost model is supplied."""

TRADING_DAYS_PER_YEAR: int = 252
"""Business days in a US equities trading year; used to annualise returns/volatility."""

TRADE_EPSILON: float = 1e-6
"""Threshold below which a share-count delta is treated as a no-trade for cost purposes."""

LOG_EPSILON: float = 1e-8
"""Small constant added inside log() to avoid log(0) in baseline strategies."""

CASH_SOFTMAX_BIAS: float = -5.0
"""Pre-softmax cash weight that maps to ~0% cash allocation when all other weights are near zero."""
