"""
Custom Gymnasium environment for portfolio optimization.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, List, Tuple, Any, Union
from .constants import (
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_TRANSACTION_COST,
    TRADE_EPSILON,
)
from .rewards import RewardFunction, SimpleReturnReward
from .transaction_costs import (
    CombinedCostModel,
    ProportionalCostModel,
    FixedSlippageModel,
    TradeInfo
)


class PortfolioEnv(gym.Env):
    """
    Multi-asset portfolio allocation environment.

    Action space: Continuous weights for each asset + cash (normalized to sum to 1 via softmax)
                  Format: [asset_1, asset_2, ..., asset_n, cash]
    Observation space: Market features + current portfolio state
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        tickers: List[str],
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        transaction_cost: float = DEFAULT_TRANSACTION_COST,
        cost_model: Optional[CombinedCostModel] = None,
        reward_function: Optional[RewardFunction] = None,
        window_size: int = 20,
        normalize_observations: bool = True,
    ):
        """
        Initialize portfolio environment.

        Args:
            data: DataFrame with MultiIndex (date, ticker) and feature columns
            feature_columns: List of column names to use as observations
            tickers: List of tickers in portfolio
            initial_balance: Starting portfolio value
            transaction_cost: Transaction cost as fraction (e.g., 0.001 = 0.1%)
                             (deprecated: use cost_model instead)
            cost_model: CombinedCostModel for transaction costs and slippage
                       If None, uses ProportionalCostModel with transaction_cost rate
            reward_function: Reward function to use (default: SimpleReturnReward)
            window_size: Number of time steps to include in observation
            normalize_observations: Whether to normalize observations
        """
        super().__init__()

        self.data = data
        self.feature_columns = feature_columns
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.initial_balance = initial_balance
        self.reward_function = reward_function or SimpleReturnReward()
        self.window_size = window_size
        self.normalize_observations = normalize_observations

        # Set up cost model (backward compatible with transaction_cost parameter)
        if cost_model is not None:
            self.cost_model = cost_model
        else:
            # Create default model from transaction_cost parameter
            self.cost_model = CombinedCostModel(
                transaction_model=ProportionalCostModel(transaction_cost),
                slippage_model=FixedSlippageModel(0.0),  # No slippage by default
                name='default'
            )

        # Keep transaction_cost for backward compatibility
        self.transaction_cost = transaction_cost

        # Get unique dates
        self.dates = data.index.get_level_values('date').unique().sort_values()
        self.n_steps = len(self.dates) - window_size

        # Validate data
        self._validate_data()

        # Define action space: portfolio weights including cash (continuous, will be normalized)
        # Using Box with finite bounds (will be passed through softmax)
        # Range [-10, 10] is sufficient for softmax input
        # Action dimensions: [asset_1, asset_2, ..., asset_n, cash]
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.n_assets + 1,),  # +1 for cash
            dtype=np.float32
        )

        # Define observation space
        # Observations: [market features (window_size x n_assets x n_features), current weights, cash ratio]
        n_features = len(feature_columns)
        obs_dim = (
            window_size * self.n_assets * n_features  # Market features
            + self.n_assets  # Current portfolio weights
            + 1  # Cash ratio
        )

        self.observation_space = spaces.Box(
            low=-1e10,
            high=1e10,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Initialize state
        self.current_step = 0
        self.portfolio_value = initial_balance
        self.cash = initial_balance
        self.holdings = np.zeros(self.n_assets)  # Number of shares
        self.weights = np.zeros(self.n_assets)  # Portfolio weights
        self.portfolio_history = []

    def _validate_data(self):
        """Validate that data contains all required tickers and features."""
        data_tickers = self.data.index.get_level_values('ticker').unique()
        for ticker in self.tickers:
            if ticker not in data_tickers:
                raise ValueError(f"Ticker {ticker} not found in data")

        for col in self.feature_columns:
            if col not in self.data.columns:
                raise ValueError(f"Feature column {col} not found in data")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        start_step: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        start_step lets callers (e.g. walk-forward backtesting) place the env
        mid-data. Must be in [0, n_steps - 1]. The observation window ends at
        dates[window_size + start_step - 1]; the first tradable bar is
        dates[window_size + start_step].
        """
        super().reset(seed=seed)

        if not 0 <= start_step < self.n_steps:
            raise ValueError(
                f"start_step={start_step} out of range [0, {self.n_steps - 1}]"
            )

        self.current_step = start_step
        self.portfolio_value = self.initial_balance
        self.cash = self.initial_balance
        self.holdings = np.zeros(self.n_assets)
        self.weights = np.zeros(self.n_assets)
        self.portfolio_history = [{
            'date': self.dates[self.window_size + start_step],
            'value': self.portfolio_value,
            'weights': self.weights.copy(),
            'cash': self.cash
        }]

        # Reset reward function state
        self.reward_function.reset()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Portfolio weight targets including cash (will be normalized via softmax)
                   Format: [asset_1_weight, ..., asset_n_weight, cash_weight]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action to valid portfolio weights using softmax
        # This includes both asset weights and cash weight
        normalized_weights = self._softmax(action)

        # Split into asset weights and cash weight
        target_weights = normalized_weights[:-1]  # Asset weights
        target_cash_ratio = normalized_weights[-1]  # Cash weight

        # Get current prices
        current_prices = self._get_prices(self.current_step)

        # Calculate current portfolio value before rebalancing
        asset_values = self.holdings * current_prices
        old_portfolio_value = self.cash + np.sum(asset_values)

        # Rebalance portfolio
        transaction_cost = self._rebalance_portfolio(
            target_weights, current_prices, old_portfolio_value, target_cash_ratio
        )

        # Move to next step
        self.current_step += 1

        # Get new prices
        new_prices = self._get_prices(self.current_step)

        # Calculate new portfolio value
        new_asset_values = self.holdings * new_prices
        new_portfolio_value = self.cash + np.sum(new_asset_values)

        # Update weights (ensure 1D array)
        if new_portfolio_value > 0:
            weights_calc = new_asset_values / new_portfolio_value
            self.weights = np.asarray(weights_calc).flatten().astype(np.float32)
        else:
            self.weights = np.zeros(self.n_assets, dtype=np.float32)

        # Calculate return
        portfolio_return = (new_portfolio_value - old_portfolio_value) / old_portfolio_value if old_portfolio_value > 0 else 0.0

        # Compute reward
        reward = self.reward_function.compute(
            portfolio_return=portfolio_return,
            portfolio_value=new_portfolio_value,
            previous_value=old_portfolio_value,
            transaction_cost=transaction_cost
        )

        # Update portfolio value
        self.portfolio_value = new_portfolio_value

        # Store history (ensure weights are 1D)
        self.portfolio_history.append({
            'date': self.dates[self.window_size + self.current_step],
            'value': self.portfolio_value,
            'weights': self.weights.flatten().copy(),  # Ensure 1D
            'cash': self.cash,
            'return': portfolio_return,
            'reward': reward,
            'transaction_cost': transaction_cost
        })

        # Check if episode is done
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        # Get new observation and info
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _rebalance_portfolio(
        self,
        target_weights: np.ndarray,
        prices: np.ndarray,
        portfolio_value: float,
        target_cash_ratio: float
    ) -> float:
        """
        Rebalance portfolio to target weights using the configured cost model.

        Args:
            target_weights: Target portfolio weights for assets (not including cash)
            prices: Current asset prices
            portfolio_value: Current total portfolio value
            target_cash_ratio: Target cash weight (0-1)

        Returns:
            Total transaction cost incurred (including slippage)
        """
        # Calculate target allocations
        # After softmax, target_weights sum to (1 - target_cash_ratio) because they're part of
        # the same softmax that includes cash. We need to renormalize them to sum to 1,
        # then multiply by the target asset value.

        # Total target value for assets (after accounting for desired cash ratio)
        target_asset_value = (1 - target_cash_ratio) * portfolio_value

        # Renormalize asset weights to sum to 1 (they represent relative proportions among assets)
        # This fixes the double application of the cash ratio
        weights_sum = np.sum(target_weights)
        if weights_sum > 0:
            normalized_asset_weights = target_weights / weights_sum
        else:
            # If all asset weights are zero, distribute equally
            normalized_asset_weights = np.ones(len(target_weights)) / len(target_weights)

        # Distribute the asset value according to the normalized weights
        target_values = normalized_asset_weights * target_asset_value

        # Calculate target number of shares
        target_holdings = target_values / prices

        # Calculate trades needed
        trades = target_holdings - self.holdings

        # Create TradeInfo objects for cost calculation
        trade_infos = []
        for i, ticker in enumerate(self.tickers):
            if abs(trades[i]) > TRADE_EPSILON:  # Only include non-trivial trades
                trade_infos.append(TradeInfo(
                    ticker=ticker,
                    shares_traded=trades[i],
                    price=prices[i],
                    portfolio_value=portfolio_value,
                    daily_volume=None,  # Could be added from data if available
                    bid_ask_spread=None  # Could be added from data if available
                ))

        # Calculate total costs using the cost model
        if trade_infos:
            cost_breakdown = self.cost_model.calculate_total_cost(trade_infos)
            total_cost = cost_breakdown['total_cost']
        else:
            total_cost = 0.0

        # Update holdings
        self.holdings = target_holdings

        # Calculate actual cash after rebalancing
        # Cash = Total portfolio - Asset value - Transaction costs
        actual_asset_value = np.sum(target_holdings * prices)
        self.cash = portfolio_value - actual_asset_value - total_cost

        return total_cost

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Get market features for the window
        features_list = []
        for i in range(self.window_size):
            step_idx = self.current_step + i
            date = self.dates[step_idx]

            for ticker in self.tickers:
                try:
                    ticker_data = self.data.xs((date, ticker), level=('date', 'ticker'))
                    # Get only the requested feature columns, fill missing with 0
                    features = []
                    for col in self.feature_columns:
                        if col in ticker_data.index:
                            features.append(ticker_data[col])
                        else:
                            features.append(0.0)
                    features = np.array(features, dtype=np.float32)
                except KeyError:
                    # If data missing, use zeros
                    features = np.zeros(len(self.feature_columns), dtype=np.float32)

                # Ensure 1D
                features = features.flatten()
                features_list.append(features)

        # Flatten all features into single 1D array
        market_features = np.concatenate(features_list)

        # Add current portfolio state
        weights_flat = self.weights.flatten() if self.weights.ndim > 1 else self.weights
        cash_ratio = np.array([self.cash / self.portfolio_value if self.portfolio_value > 0 else 1.0])

        portfolio_state = np.concatenate([
            weights_flat,  # Current weights
            cash_ratio  # Cash ratio
        ])

        # Combine all observations
        observation = np.concatenate([market_features, portfolio_state]).astype(np.float32)

        # Handle NaN/Inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return observation

    def _get_prices(self, step: int) -> np.ndarray:
        """Get asset prices at given step."""
        date = self.dates[self.window_size + step]
        prices = []

        for ticker in self.tickers:
            try:
                ticker_data = self.data.xs((date, ticker), level=('date', 'ticker'))
                price = ticker_data['close']
                # Ensure scalar value (not Series or array)
                if hasattr(price, 'item'):
                    price = price.item()
                elif hasattr(price, '__iter__') and not isinstance(price, str):
                    price = float(price[0]) if len(price) > 0 else 100.0
                else:
                    price = float(price)
            except KeyError:
                # Fall back to the previous step's price for sparse data;
                # on step 0 we have nothing to fall back to, so fail loudly
                # rather than silently inventing a price.
                if step > 0:
                    price = self._get_prices(step - 1)[self.tickers.index(ticker)]
                else:
                    raise ValueError(
                        f"Missing price for {ticker} at {date} (step 0). "
                        f"Ensure data is forward-filled before constructing the env."
                    )

            prices.append(price)

        return np.array(prices, dtype=np.float32).flatten()

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
            'cash': self.cash,
            'holdings': self.holdings.copy(),
            'date': self.dates[self.window_size + self.current_step] if self.current_step < len(self.dates) - self.window_size else None
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to convert raw actions to valid portfolio weights."""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)

    def render(self, mode: str = 'human'):
        """Render the environment state."""
        if mode == 'human':
            print(f"\nStep: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Cash: ${self.cash:.2f}")
            print("Weights:")
            for ticker, weight in zip(self.tickers, self.weights):
                print(f"  {ticker}: {weight:.2%}")

    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame."""
        return pd.DataFrame(self.portfolio_history)
