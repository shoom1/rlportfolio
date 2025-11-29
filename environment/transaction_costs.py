"""
Transaction cost and slippage models for portfolio optimization.

This module provides a flexible, pluggable system for modeling realistic
trading costs including commissions, spreads, market impact, and slippage.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class TradeInfo:
    """Information about a trade for cost calculation."""
    ticker: str
    shares_traded: float  # Positive = buy, negative = sell
    price: float
    portfolio_value: float
    daily_volume: Optional[float] = None  # For volume-based models
    bid_ask_spread: Optional[float] = None  # For spread-based models


class TransactionCostModel(ABC):
    """Base class for transaction cost models."""

    def __init__(self, name: str):
        """
        Initialize transaction cost model.

        Args:
            name: Name identifier for this model
        """
        self.name = name

    @abstractmethod
    def calculate_cost(
        self,
        trades: List[TradeInfo],
        **kwargs
    ) -> float:
        """
        Calculate transaction costs for given trades.

        Args:
            trades: List of TradeInfo objects
            **kwargs: Additional model-specific parameters

        Returns:
            Total transaction cost in dollars
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class FixedCostModel(TransactionCostModel):
    """
    Fixed cost per trade model.

    Example: $5 per trade regardless of size (like many retail brokers).
    """

    def __init__(self, cost_per_trade: float = 0.0, name: str = 'fixed'):
        """
        Initialize fixed cost model.

        Args:
            cost_per_trade: Fixed cost per trade in dollars
            name: Model identifier
        """
        super().__init__(name)
        self.cost_per_trade = cost_per_trade

    def calculate_cost(
        self,
        trades: List[TradeInfo],
        **kwargs
    ) -> float:
        """Calculate total fixed costs."""
        # Count non-zero trades
        n_trades = sum(1 for trade in trades if abs(trade.shares_traded) > 1e-6)
        return n_trades * self.cost_per_trade


class ProportionalCostModel(TransactionCostModel):
    """
    Proportional cost model (percentage of trade value).

    Example: 0.1% of trade value (common for institutional trading).
    Most commonly used model in academic literature.
    """

    def __init__(self, cost_rate: float = 0.001, name: str = 'proportional'):
        """
        Initialize proportional cost model.

        Args:
            cost_rate: Cost as fraction of trade value (e.g., 0.001 = 0.1%)
            name: Model identifier
        """
        super().__init__(name)
        self.cost_rate = cost_rate

    def calculate_cost(
        self,
        trades: List[TradeInfo],
        **kwargs
    ) -> float:
        """Calculate proportional costs."""
        total_cost = 0.0
        for trade in trades:
            trade_value = abs(trade.shares_traded * trade.price)
            total_cost += trade_value * self.cost_rate
        return total_cost


class TieredCostModel(TransactionCostModel):
    """
    Tiered cost model with volume-based pricing.

    Example:
        - First $10k: 0.5%
        - Next $90k: 0.3%
        - Above $100k: 0.1%
    """

    def __init__(
        self,
        tiers: List[tuple] = None,
        name: str = 'tiered'
    ):
        """
        Initialize tiered cost model.

        Args:
            tiers: List of (threshold, rate) tuples.
                   Example: [(10000, 0.005), (100000, 0.003), (float('inf'), 0.001)]
            name: Model identifier
        """
        super().__init__(name)

        # Default tiers if not provided
        if tiers is None:
            tiers = [
                (10000, 0.005),      # First $10k: 0.5%
                (100000, 0.003),     # Next $90k: 0.3%
                (float('inf'), 0.001) # Above $100k: 0.1%
            ]

        # Sort tiers by threshold
        self.tiers = sorted(tiers, key=lambda x: x[0])

    def calculate_cost(
        self,
        trades: List[TradeInfo],
        **kwargs
    ) -> float:
        """Calculate tiered costs."""
        # Calculate total trade value
        total_value = sum(abs(trade.shares_traded * trade.price) for trade in trades)

        # Apply tiered pricing
        total_cost = 0.0
        remaining_value = total_value
        prev_threshold = 0.0

        for threshold, rate in self.tiers:
            tier_size = min(threshold - prev_threshold, remaining_value)
            if tier_size > 0:
                total_cost += tier_size * rate
                remaining_value -= tier_size

            if remaining_value <= 0:
                break

            prev_threshold = threshold

        return total_cost


class NonlinearCostModel(TransactionCostModel):
    """
    Nonlinear market impact model.

    Cost increases nonlinearly with trade size relative to portfolio value.
    Based on market impact research: cost ∝ (trade_size)^α

    Common in academic literature:
    - Almgren & Chriss (2000): Optimal execution with nonlinear impact
    - Gârleanu & Pedersen (2013): Dynamic trading with predictable returns
    """

    def __init__(
        self,
        base_cost: float = 0.001,
        impact_coefficient: float = 0.01,
        impact_exponent: float = 1.5,
        name: str = 'nonlinear'
    ):
        """
        Initialize nonlinear cost model.

        Args:
            base_cost: Base proportional cost (e.g., 0.001 = 0.1%)
            impact_coefficient: Market impact coefficient
            impact_exponent: Exponent for market impact (typically 1.5-2.0)
            name: Model identifier
        """
        super().__init__(name)
        self.base_cost = base_cost
        self.impact_coefficient = impact_coefficient
        self.impact_exponent = impact_exponent

    def calculate_cost(
        self,
        trades: List[TradeInfo],
        **kwargs
    ) -> float:
        """Calculate nonlinear costs with market impact."""
        total_cost = 0.0

        for trade in trades:
            trade_value = abs(trade.shares_traded * trade.price)

            # Base proportional cost
            base = trade_value * self.base_cost

            # Market impact (increases nonlinearly with trade size)
            if trade.portfolio_value > 0:
                trade_fraction = trade_value / trade.portfolio_value
                impact = self.impact_coefficient * (trade_fraction ** self.impact_exponent)
                market_impact = trade_value * impact
            else:
                market_impact = 0.0

            total_cost += base + market_impact

        return total_cost


class SlippageModel(ABC):
    """Base class for slippage models."""

    def __init__(self, name: str):
        """
        Initialize slippage model.

        Args:
            name: Name identifier for this model
        """
        self.name = name

    @abstractmethod
    def calculate_slippage(
        self,
        trades: List[TradeInfo],
        **kwargs
    ) -> float:
        """
        Calculate slippage cost for given trades.

        Args:
            trades: List of TradeInfo objects
            **kwargs: Additional model-specific parameters

        Returns:
            Total slippage cost in dollars
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class FixedSlippageModel(SlippageModel):
    """
    Fixed slippage model (constant percentage).

    Example: 0.05% slippage on all trades.
    """

    def __init__(self, slippage_rate: float = 0.0005, name: str = 'fixed_slippage'):
        """
        Initialize fixed slippage model.

        Args:
            slippage_rate: Slippage as fraction of trade value (e.g., 0.0005 = 0.05%)
            name: Model identifier
        """
        super().__init__(name)
        self.slippage_rate = slippage_rate

    def calculate_slippage(
        self,
        trades: List[TradeInfo],
        **kwargs
    ) -> float:
        """Calculate fixed slippage."""
        total_slippage = 0.0
        for trade in trades:
            trade_value = abs(trade.shares_traded * trade.price)
            total_slippage += trade_value * self.slippage_rate
        return total_slippage


class VolumeBasedSlippageModel(SlippageModel):
    """
    Volume-based slippage model.

    Slippage increases when trade size is large relative to daily volume.
    Based on market microstructure research.
    """

    def __init__(
        self,
        base_slippage: float = 0.0001,
        volume_impact: float = 0.1,
        name: str = 'volume_slippage'
    ):
        """
        Initialize volume-based slippage model.

        Args:
            base_slippage: Base slippage rate
            volume_impact: Impact coefficient for volume ratio
            name: Model identifier
        """
        super().__init__(name)
        self.base_slippage = base_slippage
        self.volume_impact = volume_impact

    def calculate_slippage(
        self,
        trades: List[TradeInfo],
        **kwargs
    ) -> float:
        """Calculate volume-based slippage."""
        total_slippage = 0.0

        for trade in trades:
            trade_value = abs(trade.shares_traded * trade.price)

            # Base slippage
            slippage = self.base_slippage

            # Add volume impact if daily volume available
            if trade.daily_volume is not None and trade.daily_volume > 0:
                volume_ratio = abs(trade.shares_traded) / trade.daily_volume
                # Slippage increases with square root of volume ratio
                slippage += self.volume_impact * np.sqrt(volume_ratio)

            total_slippage += trade_value * slippage

        return total_slippage


class SpreadBasedSlippageModel(SlippageModel):
    """
    Bid-ask spread based slippage model.

    Models the cost of crossing the spread when executing trades.
    """

    def __init__(
        self,
        default_spread: float = 0.001,
        name: str = 'spread_slippage'
    ):
        """
        Initialize spread-based slippage model.

        Args:
            default_spread: Default spread as fraction of price (if not provided per trade)
            name: Model identifier
        """
        super().__init__(name)
        self.default_spread = default_spread

    def calculate_slippage(
        self,
        trades: List[TradeInfo],
        **kwargs
    ) -> float:
        """Calculate spread-based slippage."""
        total_slippage = 0.0

        for trade in trades:
            # Use provided spread or default
            spread = trade.bid_ask_spread if trade.bid_ask_spread is not None else self.default_spread

            # Crossing the spread costs half-spread on average
            # (buy at ask, sell at bid)
            trade_value = abs(trade.shares_traded * trade.price)
            total_slippage += trade_value * (spread / 2.0)

        return total_slippage


class CombinedCostModel:
    """
    Combined transaction cost and slippage model.

    Allows combining multiple cost models for realistic trading costs.
    """

    def __init__(
        self,
        transaction_model: Optional[TransactionCostModel] = None,
        slippage_model: Optional[SlippageModel] = None,
        name: str = 'combined'
    ):
        """
        Initialize combined cost model.

        Args:
            transaction_model: Transaction cost model to use
            slippage_model: Slippage model to use
            name: Model identifier
        """
        self.name = name
        self.transaction_model = transaction_model or ProportionalCostModel()
        self.slippage_model = slippage_model or FixedSlippageModel()

    def calculate_total_cost(
        self,
        trades: List[TradeInfo],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate total trading costs.

        Args:
            trades: List of TradeInfo objects
            **kwargs: Additional parameters

        Returns:
            Dictionary with breakdown of costs
        """
        transaction_cost = self.transaction_model.calculate_cost(trades, **kwargs)
        slippage_cost = self.slippage_model.calculate_slippage(trades, **kwargs)

        return {
            'transaction_cost': transaction_cost,
            'slippage': slippage_cost,
            'total_cost': transaction_cost + slippage_cost
        }

    def __repr__(self) -> str:
        return (f"CombinedCostModel(name='{self.name}', "
                f"transaction={self.transaction_model}, "
                f"slippage={self.slippage_model})")


class TransactionCostRegistry:
    """Registry for transaction cost models."""

    _models: Dict[str, TransactionCostModel] = {}
    _slippage_models: Dict[str, SlippageModel] = {}

    @classmethod
    def register_transaction_model(cls, model: TransactionCostModel):
        """Register a transaction cost model."""
        cls._models[model.name] = model

    @classmethod
    def register_slippage_model(cls, model: SlippageModel):
        """Register a slippage model."""
        cls._slippage_models[model.name] = model

    @classmethod
    def get_transaction_model(cls, name: str) -> TransactionCostModel:
        """Get transaction cost model by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown transaction cost model: {name}. "
                           f"Available: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def get_slippage_model(cls, name: str) -> SlippageModel:
        """Get slippage model by name."""
        if name not in cls._slippage_models:
            raise ValueError(f"Unknown slippage model: {name}. "
                           f"Available: {list(cls._slippage_models.keys())}")
        return cls._slippage_models[name]

    @classmethod
    def list_models(cls) -> Dict[str, List[str]]:
        """List all registered models."""
        return {
            'transaction_models': list(cls._models.keys()),
            'slippage_models': list(cls._slippage_models.keys())
        }

    @classmethod
    def create_combined_model(
        cls,
        transaction_model_name: str,
        slippage_model_name: str,
        name: str = 'combined'
    ) -> CombinedCostModel:
        """Create a combined model from registered models."""
        transaction_model = cls.get_transaction_model(transaction_model_name)
        slippage_model = cls.get_slippage_model(slippage_model_name)
        return CombinedCostModel(transaction_model, slippage_model, name)


# Register default models
def _register_defaults():
    """Register default transaction cost and slippage models."""
    # Transaction cost models
    TransactionCostRegistry.register_transaction_model(FixedCostModel(0.0, 'zero_fixed'))
    TransactionCostRegistry.register_transaction_model(FixedCostModel(1.0, 'low_fixed'))
    TransactionCostRegistry.register_transaction_model(FixedCostModel(5.0, 'standard_fixed'))

    TransactionCostRegistry.register_transaction_model(ProportionalCostModel(0.0001, 'minimal'))
    TransactionCostRegistry.register_transaction_model(ProportionalCostModel(0.001, 'standard'))
    TransactionCostRegistry.register_transaction_model(ProportionalCostModel(0.005, 'high'))

    TransactionCostRegistry.register_transaction_model(TieredCostModel(name='default_tiered'))

    TransactionCostRegistry.register_transaction_model(
        NonlinearCostModel(base_cost=0.001, impact_coefficient=0.01, name='standard_nonlinear')
    )
    TransactionCostRegistry.register_transaction_model(
        NonlinearCostModel(base_cost=0.001, impact_coefficient=0.05, name='high_impact')
    )

    # Slippage models
    TransactionCostRegistry.register_slippage_model(FixedSlippageModel(0.0, 'zero_slippage'))
    TransactionCostRegistry.register_slippage_model(FixedSlippageModel(0.0005, 'low_slippage'))
    TransactionCostRegistry.register_slippage_model(FixedSlippageModel(0.001, 'standard_slippage'))

    TransactionCostRegistry.register_slippage_model(VolumeBasedSlippageModel(name='volume_slippage'))
    TransactionCostRegistry.register_slippage_model(SpreadBasedSlippageModel(name='spread_slippage'))


# Initialize defaults
_register_defaults()


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("TRANSACTION COST MODELS DEMO")
    print("=" * 80)

    # Create sample trades
    trades = [
        TradeInfo('AAPL', 100, 150.0, 100000),
        TradeInfo('MSFT', 50, 300.0, 100000),
        TradeInfo('GOOGL', -25, 2800.0, 100000),
    ]

    print("\nSample trades:")
    for trade in trades:
        print(f"  {trade.ticker}: {trade.shares_traded:+.0f} shares @ ${trade.price:.2f}")

    # Test different models
    print("\n" + "-" * 80)
    print("TRANSACTION COST MODELS")
    print("-" * 80)

    models = [
        ('Fixed ($5/trade)', FixedCostModel(5.0)),
        ('Proportional (0.1%)', ProportionalCostModel(0.001)),
        ('Tiered', TieredCostModel()),
        ('Nonlinear (market impact)', NonlinearCostModel()),
    ]

    for name, model in models:
        cost = model.calculate_cost(trades)
        print(f"{name:30s}: ${cost:8.2f}")

    print("\n" + "-" * 80)
    print("SLIPPAGE MODELS")
    print("-" * 80)

    slippage_models = [
        ('Fixed (0.05%)', FixedSlippageModel(0.0005)),
        ('Volume-based', VolumeBasedSlippageModel()),
        ('Spread-based (0.1%)', SpreadBasedSlippageModel(0.001)),
    ]

    for name, model in slippage_models:
        cost = model.calculate_slippage(trades)
        print(f"{name:30s}: ${cost:8.2f}")

    print("\n" + "-" * 80)
    print("COMBINED MODEL")
    print("-" * 80)

    combined = CombinedCostModel(
        ProportionalCostModel(0.001),
        FixedSlippageModel(0.0005)
    )

    costs = combined.calculate_total_cost(trades)
    print(f"Transaction cost:  ${costs['transaction_cost']:8.2f}")
    print(f"Slippage:          ${costs['slippage']:8.2f}")
    print(f"Total cost:        ${costs['total_cost']:8.2f}")

    print("\n" + "-" * 80)
    print("REGISTERED MODELS")
    print("-" * 80)

    available = TransactionCostRegistry.list_models()
    print(f"\nTransaction models: {', '.join(available['transaction_models'])}")
    print(f"Slippage models: {', '.join(available['slippage_models'])}")

    # Test registry
    print("\n" + "-" * 80)
    print("USING REGISTRY")
    print("-" * 80)

    model = TransactionCostRegistry.get_transaction_model('standard')
    cost = model.calculate_cost(trades)
    print(f"Standard model cost: ${cost:.2f}")

    combined_from_registry = TransactionCostRegistry.create_combined_model(
        'standard', 'low_slippage', 'realistic'
    )
    costs = combined_from_registry.calculate_total_cost(trades)
    print(f"Combined from registry: ${costs['total_cost']:.2f}")

    print("\n" + "=" * 80)
