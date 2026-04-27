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
                # Explicit zero-slippage default for backward compatibility.
                # Use a custom cost_model to enable nonzero or data-aware slippage.
                slippage_model=FixedSlippageModel(0.0),
                name='default'
            )

        # Keep transaction_cost for backward compatibility
        self.transaction_cost = transaction_cost

        # Get unique dates
        self.dates = data.index.get_level_values('date').unique().sort_values()
        self.n_steps = len(self.dates) - window_size

        # Validate data
        self._validate_data()

        # Precompute dense (n_dates, n_tickers, n_features) and (n_dates,
        # n_tickers) arrays so the hot path never touches pandas MultiIndex
        # lookups. Profiling showed xs() + datetime parsing was ~96% of the
        # per-step cost at 4 fps.
        self._precompute_arrays()

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

        # Initialize state. weights/holdings are always 1-D float arrays;
        # the shape is an invariant relied on by _get_observation.
        self.current_step = 0
        self.portfolio_value = initial_balance
        self.cash = initial_balance
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
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

    def _precompute_arrays(self):
        """Materialise the data into dense numpy cubes indexed by
        (date_idx, ticker_idx, ...), so step-time feature/price lookups are
        O(1) slices instead of pandas MultiIndex xs() calls.

        Behavioural equivalence with the previous xs()-based path:
          - Features: missing (date, ticker) rows fill with 0 (the old
            KeyError branch did the same).
          - Prices: forward-filled per ticker so a missing row falls back to
            the most recent known price (the old recursive previous-step
            fallback). Leading NaN is left as-is; _get_prices raises on it,
            matching the old step-0 ValueError.
        """
        index = pd.MultiIndex.from_product(
            [self.dates, self.tickers], names=['date', 'ticker']
        )

        feat_df = self.data[self.feature_columns].reindex(index).fillna(0.0)
        self._feature_array = (
            feat_df.to_numpy(dtype=np.float32, copy=False)
            .reshape(len(self.dates), self.n_assets, len(self.feature_columns))
        )

        close = self.data[['close']].reindex(index)
        close = close.groupby(level='ticker').ffill()
        self._price_array = (
            close['close']
            .to_numpy(dtype=np.float32, copy=False)
            .reshape(len(self.dates), self.n_assets)
        )

        self._volume_array = self._precompute_optional_market_array(index, 'volume')

        spread_col = next(
            (
                col for col in ('bid_ask_spread', 'spread')
                if col in self.data.columns
            ),
            None,
        )
        self._spread_array = (
            self._precompute_optional_market_array(index, spread_col)
            if spread_col is not None
            else None
        )

    def _precompute_optional_market_array(
        self,
        index: pd.MultiIndex,
        column: Optional[str],
    ) -> Optional[np.ndarray]:
        if column is None or column not in self.data.columns:
            return None

        values = self.data[[column]].reindex(index)
        values = values.groupby(level='ticker').ffill()
        return (
            values[column]
            .to_numpy(dtype=np.float64, copy=False)
            .reshape(len(self.dates), self.n_assets)
        )

    def _get_optional_market_value(
        self,
        values: Optional[np.ndarray],
        date_idx: int,
        asset_idx: int,
        *,
        require_positive: bool = False,
        require_nonnegative: bool = False,
    ) -> Optional[float]:
        if values is None:
            return None
        value = float(values[date_idx, asset_idx])
        if not np.isfinite(value):
            return None
        if require_positive and value <= 0:
            return None
        if require_nonnegative and value < 0:
            return None
        return value

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
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
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

        # Insolvency: if the portfolio has been wiped out, terminate the
        # episode with a full-loss return so the reward reflects ruin
        # rather than a silent 0.0 that keeps the agent stepping through
        # garbage observations.
        bankrupt = new_portfolio_value <= 0

        if bankrupt:
            self.weights = np.zeros(self.n_assets, dtype=np.float32)
            portfolio_return = -1.0
        else:
            self.weights = (new_asset_values / new_portfolio_value).astype(np.float32)
            portfolio_return = (
                (new_portfolio_value - old_portfolio_value) / old_portfolio_value
                if old_portfolio_value > 0 else 0.0
            )

        # Compute reward
        reward = self.reward_function.compute(
            portfolio_return=portfolio_return,
            portfolio_value=new_portfolio_value,
            previous_value=old_portfolio_value,
            transaction_cost=transaction_cost
        )

        # Update portfolio value
        self.portfolio_value = new_portfolio_value

        # Store history
        self.portfolio_history.append({
            'date': self.dates[self.window_size + self.current_step],
            'value': self.portfolio_value,
            'weights': self.weights.copy(),
            'cash': self.cash,
            'return': portfolio_return,
            'reward': reward,
            'transaction_cost': transaction_cost
        })

        # Check if episode is done
        terminated = bankrupt or self.current_step >= self.n_steps - 1
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
        if portfolio_value <= 0:
            self.holdings = np.zeros(self.n_assets, dtype=np.float64)
            self.cash = portfolio_value
            return 0.0

        # Renormalize asset weights to sum to 1 (they represent relative proportions among assets)
        # This fixes the double application of the cash ratio
        weights_sum = np.sum(target_weights)
        if weights_sum > 0:
            normalized_asset_weights = target_weights / weights_sum
        else:
            # If all asset weights are zero, distribute equally
            normalized_asset_weights = np.ones(len(target_weights)) / len(target_weights)

        target_cash_ratio = float(np.clip(target_cash_ratio, 0.0, 1.0))
        current_holdings = self.holdings.copy()
        market_date_idx = self.window_size + self.current_step

        def build_trade_infos(candidate_holdings: np.ndarray) -> List[TradeInfo]:
            trades = candidate_holdings - current_holdings
            trade_infos = []
            for i, ticker in enumerate(self.tickers):
                if abs(trades[i]) > TRADE_EPSILON:
                    trade_infos.append(TradeInfo(
                        ticker=ticker,
                        shares_traded=trades[i],
                        price=prices[i],
                        portfolio_value=portfolio_value,
                        daily_volume=self._get_optional_market_value(
                            self._volume_array,
                            market_date_idx,
                            i,
                            require_positive=True,
                        ),
                        bid_ask_spread=self._get_optional_market_value(
                            self._spread_array,
                            market_date_idx,
                            i,
                            require_nonnegative=True,
                        ),
                    ))
            return trade_infos

        def calculate_cost(candidate_holdings: np.ndarray) -> float:
            trade_infos = build_trade_infos(candidate_holdings)
            if not trade_infos:
                return 0.0
            cost_breakdown = self.cost_model.calculate_total_cost(trade_infos)
            return float(cost_breakdown['total_cost'])

        target_holdings = self.holdings.copy()
        total_cost = 0.0
        tolerance = max(1e-8, abs(portfolio_value) * 1e-10)

        # Transaction costs are paid from the same starting portfolio. Size
        # target assets against post-cost value so low-cash actions cannot
        # borrow implicitly to pay costs.
        for _ in range(25):
            post_cost_value = max(portfolio_value - total_cost, 0.0)
            target_asset_value = (1.0 - target_cash_ratio) * post_cost_value
            target_values = normalized_asset_weights * target_asset_value
            candidate_holdings = target_values / prices
            candidate_cost = calculate_cost(candidate_holdings)

            target_holdings = candidate_holdings
            if abs(candidate_cost - total_cost) <= tolerance:
                total_cost = candidate_cost
                break
            total_cost = candidate_cost

        # Calculate actual cash after rebalancing
        # Cash = total portfolio - asset value - transaction costs. Clamp
        # tiny negative roundoff to zero; material negative cash would imply
        # margin, which this environment does not model.
        actual_asset_value = np.sum(target_holdings * prices)
        cash = portfolio_value - actual_asset_value - total_cost

        if cash < -tolerance:
            best_holdings = current_holdings
            best_cost = 0.0
            low, high = 0.0, 1.0
            for _ in range(60):
                alpha = (low + high) / 2.0
                candidate_holdings = (
                    current_holdings + alpha * (target_holdings - current_holdings)
                )
                candidate_cost = calculate_cost(candidate_holdings)
                candidate_asset_value = np.sum(candidate_holdings * prices)
                candidate_cash = (
                    portfolio_value - candidate_asset_value - candidate_cost
                )
                if candidate_cash >= 0:
                    best_holdings = candidate_holdings
                    best_cost = candidate_cost
                    low = alpha
                else:
                    high = alpha

            target_holdings = best_holdings
            total_cost = best_cost
            actual_asset_value = np.sum(target_holdings * prices)
            cash = portfolio_value - actual_asset_value - total_cost

        # Update holdings and cash together after all cost calculations have
        # been made against the original holdings.
        self.holdings = target_holdings
        self.cash = 0.0 if -tolerance < cash < 0 else cash

        return total_cost

    def _get_observation(self) -> np.ndarray:
        """Get current observation.

        Hot path — must not touch pandas MultiIndex. Reads a contiguous
        slice of the precomputed feature cube and appends portfolio state.
        """
        start = self.current_step
        end = start + self.window_size
        market_features = self._feature_array[start:end].reshape(-1)

        cash_ratio = np.array(
            [self.cash / self.portfolio_value if self.portfolio_value > 0 else 1.0]
        )
        portfolio_state = np.concatenate([self.weights, cash_ratio])

        observation = np.concatenate([market_features, portfolio_state]).astype(np.float32)

        # Handle NaN/Inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return observation

    def _get_prices(self, step: int) -> np.ndarray:
        """Get asset prices at the given step.

        Reads from the precomputed, per-ticker forward-filled price array.
        Leading NaN (no prior price to fall back to) raises, matching the
        previous step-0 ValueError behavior.
        """
        date_idx = self.window_size + step
        prices = self._price_array[date_idx]
        if np.any(np.isnan(prices)):
            missing = [t for t, p in zip(self.tickers, prices) if np.isnan(p)]
            date = self.dates[date_idx]
            raise ValueError(
                f"Missing price for {missing} at {date} (step {step}). "
                f"Ensure data is forward-filled before constructing the env."
            )
        return prices

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
