"""
Examples demonstrating different transaction cost and slippage models.

This script shows how to:
1. Use different cost models in the PortfolioEnv
2. Compare performance under different cost assumptions
3. Create custom cost models
4. Use the cost model registry
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from environment.transaction_costs import (
    ProportionalCostModel,
    TieredCostModel,
    NonlinearCostModel,
    FixedSlippageModel,
    VolumeBasedSlippageModel,
    SpreadBasedSlippageModel,
    CombinedCostModel,
    TransactionCostRegistry,
    TradeInfo
)


def example_1_basic_cost_models():
    """Example 1: Compare different transaction cost models."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Comparing Transaction Cost Models")
    print("="*80)

    # Sample trades
    trades = [
        TradeInfo('AAPL', 100, 175.0, 100000),
        TradeInfo('MSFT', 50, 380.0, 100000),
        TradeInfo('GOOGL', -25, 140.0, 100000),
    ]

    total_value = sum(abs(t.shares_traded * t.price) for t in trades)
    print(f"\nTotal trade value: ${total_value:,.2f}")
    print("\nCost by model:")

    models = {
        'Low Cost (0.05%)': ProportionalCostModel(0.0005),
        'Standard (0.1%)': ProportionalCostModel(0.001),
        'High Cost (0.5%)': ProportionalCostModel(0.005),
        'Tiered Pricing': TieredCostModel(),
        'Nonlinear (Market Impact)': NonlinearCostModel(
            base_cost=0.001,
            impact_coefficient=0.01
        )
    }

    for name, model in models.items():
        cost = model.calculate_cost(trades)
        pct = (cost / total_value) * 100
        print(f"  {name:30s}: ${cost:8.2f} ({pct:.3f}%)")


def example_2_slippage_models():
    """Example 2: Compare different slippage models."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Comparing Slippage Models")
    print("="*80)

    # Trades with volume and spread data
    trades = [
        TradeInfo('AAPL', 10000, 175.0, 100000,
                 daily_volume=50000000, bid_ask_spread=0.01),
        TradeInfo('MSFT', 5000, 380.0, 100000,
                 daily_volume=30000000, bid_ask_spread=0.015),
        TradeInfo('GOOGL', -2500, 140.0, 100000,
                 daily_volume=20000000, bid_ask_spread=0.02),
    ]

    total_value = sum(abs(t.shares_traded * t.price) for t in trades)
    print(f"\nTotal trade value: ${total_value:,.2f}")
    print("\nSlippage by model:")

    models = {
        'Fixed (0.05%)': FixedSlippageModel(0.0005),
        'Volume-Based': VolumeBasedSlippageModel(
            base_slippage=0.0001,
            volume_impact=0.1
        ),
        'Spread-Based': SpreadBasedSlippageModel()
    }

    for name, model in models.items():
        cost = model.calculate_slippage(trades)
        pct = (cost / total_value) * 100
        print(f"  {name:30s}: ${cost:8.2f} ({pct:.3f}%)")


def example_3_combined_models():
    """Example 3: Realistic combined cost models."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Combined Transaction Costs + Slippage")
    print("="*80)

    trades = [
        TradeInfo('AAPL', 5000, 175.0, 500000,
                 daily_volume=50000000, bid_ask_spread=0.01),
        TradeInfo('MSFT', 2000, 380.0, 500000,
                 daily_volume=30000000, bid_ask_spread=0.015),
    ]

    total_value = sum(abs(t.shares_traded * t.price) for t in trades)
    print(f"\nTotal trade value: ${total_value:,.2f}")
    print("\nScenarios:")

    scenarios = {
        'Retail Broker (Low Cost)': CombinedCostModel(
            ProportionalCostModel(0.0),  # Commission-free
            FixedSlippageModel(0.001),   # 0.1% slippage
            'retail'
        ),
        'Institutional (Standard)': CombinedCostModel(
            ProportionalCostModel(0.001),  # 0.1% commission
            VolumeBasedSlippageModel(),
            'institutional'
        ),
        'High Frequency (Nonlinear)': CombinedCostModel(
            NonlinearCostModel(base_cost=0.0005, impact_coefficient=0.05),
            SpreadBasedSlippageModel(),
            'hft'
        )
    }

    for name, model in scenarios.items():
        costs = model.calculate_total_cost(trades)
        pct = (costs['total_cost'] / total_value) * 100

        print(f"\n{name}:")
        print(f"  Transaction cost: ${costs['transaction_cost']:8.2f}")
        print(f"  Slippage:         ${costs['slippage']:8.2f}")
        print(f"  Total cost:       ${costs['total_cost']:8.2f} ({pct:.3f}%)")


def example_4_using_registry():
    """Example 4: Using the cost model registry."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Using the Transaction Cost Registry")
    print("="*80)

    # List available models
    available = TransactionCostRegistry.list_models()
    print("\nAvailable transaction cost models:")
    for model_name in available['transaction_models']:
        print(f"  - {model_name}")

    print("\nAvailable slippage models:")
    for model_name in available['slippage_models']:
        print(f"  - {model_name}")

    # Create combined model from registry
    print("\nCreating combined model from registry...")
    combined = TransactionCostRegistry.create_combined_model(
        'standard',      # 0.1% proportional cost
        'low_slippage',  # 0.05% slippage
        'my_model'
    )

    trades = [TradeInfo('AAPL', 1000, 175.0, 100000)]
    costs = combined.calculate_total_cost(trades)

    print(f"Model: {combined.name}")
    print(f"Total cost: ${costs['total_cost']:.2f}")


def example_5_custom_cost_model():
    """Example 5: Creating custom cost models."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Creating Custom Cost Models")
    print("="*80)

    # Custom tiered pricing for large traders
    custom_tiers = [
        (50000, 0.002),      # First $50k: 0.2%
        (500000, 0.001),     # Next $450k: 0.1%
        (float('inf'), 0.0005)  # Above $500k: 0.05%
    ]

    custom_model = TieredCostModel(custom_tiers, name='large_trader')

    # Register it
    TransactionCostRegistry.register_transaction_model(custom_model)

    print("\nCustom model registered as 'large_trader'")
    print("Tier structure:")
    for threshold, rate in custom_tiers:
        if threshold == float('inf'):
            print(f"  Above previous: {rate*100}%")
        else:
            print(f"  Up to ${threshold:,}: {rate*100}%")

    # Test it
    trades = [TradeInfo('AAPL', 10000, 100.0, 5000000)]  # $1M trade
    cost = custom_model.calculate_cost(trades)

    print(f"\nCost for $1M trade: ${cost:,.2f}")

    # Clean up
    del TransactionCostRegistry._models['large_trader']


def example_6_environment_integration():
    """Example 6: Using cost models in PortfolioEnv."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Integration with PortfolioEnv")
    print("="*80)

    print("""
To use custom cost models in PortfolioEnv:

from environment.portfolio_env import PortfolioEnv
from environment.transaction_costs import (
    CombinedCostModel,
    NonlinearCostModel,
    VolumeBasedSlippageModel
)

# Create a realistic cost model
cost_model = CombinedCostModel(
    transaction_model=NonlinearCostModel(
        base_cost=0.001,        # 0.1% base cost
        impact_coefficient=0.01, # Market impact
        impact_exponent=1.5
    ),
    slippage_model=VolumeBasedSlippageModel(
        base_slippage=0.0001,
        volume_impact=0.1
    ),
    name='realistic'
)

# Use it in the environment
env = PortfolioEnv(
    data=your_data,
    feature_columns=feature_cols,
    tickers=tickers,
    initial_balance=100000,
    cost_model=cost_model  # Pass the custom model
)

# The environment will now use your cost model for all trades!

# For backward compatibility, you can still use the simple parameter:
env = PortfolioEnv(
    ...
    transaction_cost=0.001  # Creates ProportionalCostModel(0.001)
)
    """)


def example_7_cost_comparison_backtest():
    """Example 7: Compare strategy performance under different costs."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Impact of Costs on Strategy Performance")
    print("="*80)

    print("""
Typical cost scenarios and their impact:

1. ZERO COSTS (Unrealistic)
   - Transaction: 0%
   - Slippage: 0%
   - Use case: Theoretical testing only
   - Impact: Overestimates returns by 0.5-2% annually

2. LOW COSTS (Large institutions, commission-free brokers)
   - Transaction: 0-0.05%
   - Slippage: 0.01-0.05%
   - Use case: Institutional trading, ETFs
   - Impact: Minimal on low-frequency strategies

3. STANDARD COSTS (Typical retail)
   - Transaction: 0.1%
   - Slippage: 0.05-0.1%
   - Use case: Most realistic for retail/small funds
   - Impact: 0.2-1% annual drag depending on turnover

4. HIGH COSTS (Small broker, illiquid assets)
   - Transaction: 0.5%
   - Slippage: 0.2-0.5%
   - Use case: Illiquid markets, small brokers
   - Impact: 1-3% annual drag, makes HFT unprofitable

5. NONLINEAR (Market impact matters)
   - Transaction: 0.1% + impact
   - Slippage: Volume-dependent
   - Use case: Large trades, thin markets
   - Impact: Punishes large position changes

Recommendation: Use STANDARD or NONLINEAR for realistic backtesting.
    """)

    # Simulate impact on returns
    print("\nSimulated annual impact by strategy type:")
    print("-" * 60)

    strategies = [
        ('Buy & Hold', 2, 0.001),      # 2 rebalances/year
        ('Monthly Rebalance', 12, 0.001),
        ('Momentum (Weekly)', 52, 0.001),
        ('High Frequency', 250, 0.001),
    ]

    for name, trades_per_year, avg_cost in strategies:
        annual_drag = trades_per_year * avg_cost * 100
        print(f"{name:25s}: -{annual_drag:.2f}% per year")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("TRANSACTION COST AND SLIPPAGE MODELS - EXAMPLES")
    print("="*80)

    example_1_basic_cost_models()
    example_2_slippage_models()
    example_3_combined_models()
    example_4_using_registry()
    example_5_custom_cost_model()
    example_6_environment_integration()
    example_7_cost_comparison_backtest()

    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. Transaction costs SIGNIFICANTLY impact strategy performance
2. Different models for different use cases:
   - Proportional: Simple, most common
   - Tiered: Volume discounts
   - Nonlinear: Market impact for large trades
3. Don't forget slippage! Often as important as commissions
4. Use realistic costs for backtesting (0.1-0.2% total is typical)
5. High-frequency strategies require very low costs to be profitable
6. Cost models are pluggable - easy to experiment
7. Always test with MULTIPLE cost assumptions

Happy (realistic) backtesting! 📈
    """)
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
