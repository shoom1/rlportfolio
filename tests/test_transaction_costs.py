"""
Unit tests for transaction cost and slippage models.
"""

import pytest
import numpy as np
from environment.transaction_costs import (
    TradeInfo,
    FixedCostModel,
    ProportionalCostModel,
    TieredCostModel,
    NonlinearCostModel,
    FixedSlippageModel,
    VolumeBasedSlippageModel,
    SpreadBasedSlippageModel,
    CombinedCostModel,
    TransactionCostRegistry
)


@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    return [
        TradeInfo('AAPL', 100, 150.0, 100000, daily_volume=1000000, bid_ask_spread=0.01),
        TradeInfo('MSFT', 50, 300.0, 100000, daily_volume=800000, bid_ask_spread=0.015),
        TradeInfo('GOOGL', -25, 2800.0, 100000, daily_volume=500000, bid_ask_spread=0.02),
    ]


@pytest.fixture
def no_trades():
    """Create empty trades list."""
    return []


@pytest.fixture
def single_trade():
    """Create single trade for testing."""
    return [TradeInfo('AAPL', 100, 150.0, 100000)]


# ============================================================================
# Fixed Cost Model Tests
# ============================================================================

class TestFixedCostModel:
    """Tests for FixedCostModel."""

    def test_zero_cost(self, sample_trades):
        """Test fixed model with zero cost."""
        model = FixedCostModel(cost_per_trade=0.0)
        cost = model.calculate_cost(sample_trades)
        assert cost == 0.0

    def test_standard_cost(self, sample_trades):
        """Test fixed cost per trade."""
        model = FixedCostModel(cost_per_trade=5.0)
        cost = model.calculate_cost(sample_trades)
        # 3 trades at $5 each
        assert cost == 15.0

    def test_no_trades(self, no_trades):
        """Test with no trades."""
        model = FixedCostModel(cost_per_trade=5.0)
        cost = model.calculate_cost(no_trades)
        assert cost == 0.0

    def test_single_trade(self, single_trade):
        """Test with single trade."""
        model = FixedCostModel(cost_per_trade=10.0)
        cost = model.calculate_cost(single_trade)
        assert cost == 10.0

    def test_name_attribute(self):
        """Test model has correct name."""
        model = FixedCostModel(5.0, name='my_fixed')
        assert model.name == 'my_fixed'


# ============================================================================
# Proportional Cost Model Tests
# ============================================================================

class TestProportionalCostModel:
    """Tests for ProportionalCostModel."""

    def test_standard_rate(self, sample_trades):
        """Test proportional cost calculation."""
        model = ProportionalCostModel(cost_rate=0.001)  # 0.1%

        # Calculate expected cost
        # Trade 1: 100 * 150 = 15,000 -> 15
        # Trade 2: 50 * 300 = 15,000 -> 15
        # Trade 3: 25 * 2800 = 70,000 -> 70
        # Total: 100
        expected = (100 * 150 + 50 * 300 + 25 * 2800) * 0.001

        cost = model.calculate_cost(sample_trades)
        assert abs(cost - expected) < 1e-6

    def test_zero_rate(self, sample_trades):
        """Test with zero cost rate."""
        model = ProportionalCostModel(cost_rate=0.0)
        cost = model.calculate_cost(sample_trades)
        assert cost == 0.0

    def test_high_rate(self, single_trade):
        """Test with high cost rate."""
        model = ProportionalCostModel(cost_rate=0.01)  # 1%
        cost = model.calculate_cost(single_trade)
        # 100 * 150 * 0.01 = 150
        assert abs(cost - 150.0) < 1e-6

    def test_no_trades(self, no_trades):
        """Test with no trades."""
        model = ProportionalCostModel(cost_rate=0.001)
        cost = model.calculate_cost(no_trades)
        assert cost == 0.0

    def test_absolute_value(self):
        """Test that negative shares (sells) use absolute value."""
        trades = [TradeInfo('AAPL', -100, 150.0, 100000)]
        model = ProportionalCostModel(cost_rate=0.001)
        cost = model.calculate_cost(trades)
        # Should be same as buying 100 shares
        assert abs(cost - 15.0) < 1e-6


# ============================================================================
# Tiered Cost Model Tests
# ============================================================================

class TestTieredCostModel:
    """Tests for TieredCostModel."""

    def test_default_tiers(self, single_trade):
        """Test with default tier structure."""
        model = TieredCostModel()
        # Trade value: 100 * 150 = 15,000
        # Default tiers: [(10000, 0.005), (100000, 0.003), (inf, 0.001)]
        # First 10k at 0.5%: 10,000 * 0.005 = 50
        # Next 5k at 0.3%: 5,000 * 0.003 = 15
        # Expected: 65
        cost = model.calculate_cost(single_trade)
        expected = 10000 * 0.005 + 5000 * 0.003
        assert abs(cost - expected) < 1e-6

    def test_multiple_tiers(self):
        """Test trade spanning multiple tiers."""
        # Custom tiers: [10k @ 0.5%, 90k @ 0.3%, rest @ 0.1%]
        model = TieredCostModel([
            (10000, 0.005),
            (100000, 0.003),
            (float('inf'), 0.001)
        ])

        # Trade worth $150,000
        trades = [TradeInfo('AAPL', 1000, 150.0, 1000000)]
        cost = model.calculate_cost(trades)

        # First 10k at 0.5%: 10,000 * 0.005 = 50
        # Next 90k at 0.3%: 90,000 * 0.003 = 270
        # Remaining 50k at 0.1%: 50,000 * 0.001 = 50
        expected = 50 + 270 + 50
        assert abs(cost - expected) < 1e-6

    def test_custom_tiers(self):
        """Test with custom tier structure."""
        tiers = [
            (5000, 0.01),   # First $5k: 1%
            (20000, 0.005), # Next $15k: 0.5%
            (float('inf'), 0.002)  # Rest: 0.2%
        ]
        model = TieredCostModel(tiers)

        trades = [TradeInfo('AAPL', 100, 100.0, 50000)]  # $10k trade
        cost = model.calculate_cost(trades)

        # First 5k at 1%: 50
        # Next 5k at 0.5%: 25
        expected = 50 + 25
        assert abs(cost - expected) < 1e-6

    def test_no_trades(self, no_trades):
        """Test with no trades."""
        model = TieredCostModel()
        cost = model.calculate_cost(no_trades)
        assert cost == 0.0


# ============================================================================
# Nonlinear Cost Model Tests
# ============================================================================

class TestNonlinearCostModel:
    """Tests for NonlinearCostModel."""

    def test_base_cost_only(self):
        """Test with zero impact coefficient."""
        model = NonlinearCostModel(
            base_cost=0.001,
            impact_coefficient=0.0,  # No market impact
            impact_exponent=1.5
        )

        trades = [TradeInfo('AAPL', 100, 150.0, 100000)]
        cost = model.calculate_cost(trades)

        # Should be just base cost
        expected = 100 * 150 * 0.001
        assert abs(cost - expected) < 1e-6

    def test_with_market_impact(self):
        """Test market impact increases cost."""
        model = NonlinearCostModel(
            base_cost=0.001,
            impact_coefficient=0.01,
            impact_exponent=1.5
        )

        trades = [TradeInfo('AAPL', 100, 150.0, 100000)]
        cost = model.calculate_cost(trades)

        # Should be greater than base cost alone
        base_only = 100 * 150 * 0.001
        assert cost > base_only

    def test_larger_trade_higher_impact(self):
        """Test that larger trades have higher impact."""
        model = NonlinearCostModel(
            base_cost=0.001,
            impact_coefficient=0.01,
            impact_exponent=1.5
        )

        # Small trade
        small_trade = [TradeInfo('AAPL', 10, 150.0, 100000)]
        small_cost = model.calculate_cost(small_trade)

        # Large trade (10x)
        large_trade = [TradeInfo('AAPL', 100, 150.0, 100000)]
        large_cost = model.calculate_cost(large_trade)

        # Cost should increase more than linearly
        assert large_cost > small_cost * 10

    def test_no_trades(self, no_trades):
        """Test with no trades."""
        model = NonlinearCostModel()
        cost = model.calculate_cost(no_trades)
        assert cost == 0.0

    def test_zero_portfolio_value(self):
        """Test with zero portfolio value (edge case)."""
        model = NonlinearCostModel()
        trades = [TradeInfo('AAPL', 100, 150.0, 0.0)]
        cost = model.calculate_cost(trades)
        # Should not crash, returns base cost only
        assert cost >= 0.0


# ============================================================================
# Fixed Slippage Model Tests
# ============================================================================

class TestFixedSlippageModel:
    """Tests for FixedSlippageModel."""

    def test_standard_slippage(self, sample_trades):
        """Test fixed slippage calculation."""
        model = FixedSlippageModel(slippage_rate=0.0005)  # 0.05%

        expected = (100 * 150 + 50 * 300 + 25 * 2800) * 0.0005
        cost = model.calculate_slippage(sample_trades)

        assert abs(cost - expected) < 1e-6

    def test_zero_slippage(self, sample_trades):
        """Test with zero slippage."""
        model = FixedSlippageModel(slippage_rate=0.0)
        cost = model.calculate_slippage(sample_trades)
        assert cost == 0.0

    def test_no_trades(self, no_trades):
        """Test with no trades."""
        model = FixedSlippageModel(slippage_rate=0.001)
        cost = model.calculate_slippage(no_trades)
        assert cost == 0.0


# ============================================================================
# Volume-Based Slippage Model Tests
# ============================================================================

class TestVolumeBasedSlippageModel:
    """Tests for VolumeBasedSlippageModel."""

    def test_with_volume_data(self, sample_trades):
        """Test slippage with volume data."""
        model = VolumeBasedSlippageModel(
            base_slippage=0.0001,
            volume_impact=0.1
        )

        cost = model.calculate_slippage(sample_trades)
        # Should include both base slippage and volume impact
        assert cost > 0.0

    def test_without_volume_data(self):
        """Test slippage without volume data (falls back to base)."""
        trades = [TradeInfo('AAPL', 100, 150.0, 100000, daily_volume=None)]
        model = VolumeBasedSlippageModel(base_slippage=0.0001)

        cost = model.calculate_slippage(trades)
        # Should be just base slippage
        expected = 100 * 150 * 0.0001
        assert abs(cost - expected) < 1e-6

    def test_large_volume_ratio(self):
        """Test that large trades relative to volume have higher slippage."""
        model = VolumeBasedSlippageModel(base_slippage=0.0001, volume_impact=0.1)

        # Small trade relative to volume
        small = [TradeInfo('AAPL', 100, 150.0, 100000, daily_volume=1000000)]
        small_cost = model.calculate_slippage(small)

        # Large trade relative to volume
        large = [TradeInfo('AAPL', 100000, 150.0, 100000000, daily_volume=1000000)]
        large_cost = model.calculate_slippage(large)

        # Large trade should have proportionally higher slippage
        assert large_cost > small_cost * 100

    def test_no_trades(self, no_trades):
        """Test with no trades."""
        model = VolumeBasedSlippageModel()
        cost = model.calculate_slippage(no_trades)
        assert cost == 0.0


# ============================================================================
# Spread-Based Slippage Model Tests
# ============================================================================

class TestSpreadBasedSlippageModel:
    """Tests for SpreadBasedSlippageModel."""

    def test_with_spread_data(self, sample_trades):
        """Test slippage with bid-ask spread data."""
        model = SpreadBasedSlippageModel(default_spread=0.001)
        cost = model.calculate_slippage(sample_trades)

        # Should use provided spreads from TradeInfo
        # Cost is half-spread on trade value
        expected = (
            100 * 150 * 0.01 / 2 +
            50 * 300 * 0.015 / 2 +
            25 * 2800 * 0.02 / 2
        )
        assert abs(cost - expected) < 1e-6

    def test_without_spread_data(self):
        """Test slippage without spread data (uses default)."""
        trades = [TradeInfo('AAPL', 100, 150.0, 100000, bid_ask_spread=None)]
        model = SpreadBasedSlippageModel(default_spread=0.002)

        cost = model.calculate_slippage(trades)
        # Should use default spread
        expected = 100 * 150 * 0.002 / 2
        assert abs(cost - expected) < 1e-6

    def test_wide_spread_higher_cost(self):
        """Test that wider spreads result in higher costs."""
        model = SpreadBasedSlippageModel()

        narrow = [TradeInfo('AAPL', 100, 150.0, 100000, bid_ask_spread=0.001)]
        narrow_cost = model.calculate_slippage(narrow)

        wide = [TradeInfo('AAPL', 100, 150.0, 100000, bid_ask_spread=0.01)]
        wide_cost = model.calculate_slippage(wide)

        assert wide_cost > narrow_cost

    def test_no_trades(self, no_trades):
        """Test with no trades."""
        model = SpreadBasedSlippageModel()
        cost = model.calculate_slippage(no_trades)
        assert cost == 0.0


# ============================================================================
# Combined Cost Model Tests
# ============================================================================

class TestCombinedCostModel:
    """Tests for CombinedCostModel."""

    def test_combined_calculation(self, sample_trades):
        """Test combined transaction cost and slippage."""
        transaction_model = ProportionalCostModel(0.001)
        slippage_model = FixedSlippageModel(0.0005)

        combined = CombinedCostModel(transaction_model, slippage_model)
        costs = combined.calculate_total_cost(sample_trades)

        # Check all cost components are present
        assert 'transaction_cost' in costs
        assert 'slippage' in costs
        assert 'total_cost' in costs

        # Total should equal sum
        assert abs(costs['total_cost'] - (costs['transaction_cost'] + costs['slippage'])) < 1e-6

    def test_default_models(self, sample_trades):
        """Test combined model with defaults."""
        combined = CombinedCostModel()
        costs = combined.calculate_total_cost(sample_trades)

        assert costs['total_cost'] >= 0.0

    def test_custom_models(self, sample_trades):
        """Test with custom transaction and slippage models."""
        transaction_model = NonlinearCostModel(base_cost=0.001, impact_coefficient=0.01)
        slippage_model = VolumeBasedSlippageModel()

        combined = CombinedCostModel(transaction_model, slippage_model, name='custom')
        costs = combined.calculate_total_cost(sample_trades)

        assert costs['total_cost'] > 0.0
        assert combined.name == 'custom'

    def test_no_trades(self, no_trades):
        """Test with no trades."""
        combined = CombinedCostModel()
        costs = combined.calculate_total_cost(no_trades)

        assert costs['transaction_cost'] == 0.0
        assert costs['slippage'] == 0.0
        assert costs['total_cost'] == 0.0


# ============================================================================
# Transaction Cost Registry Tests
# ============================================================================

class TestTransactionCostRegistry:
    """Tests for TransactionCostRegistry."""

    def test_get_transaction_model(self):
        """Test retrieving registered transaction model."""
        model = TransactionCostRegistry.get_transaction_model('standard')
        assert isinstance(model, ProportionalCostModel)
        assert model.cost_rate == 0.001

    def test_get_slippage_model(self):
        """Test retrieving registered slippage model."""
        model = TransactionCostRegistry.get_slippage_model('low_slippage')
        assert isinstance(model, FixedSlippageModel)
        assert model.slippage_rate == 0.0005

    def test_unknown_model_raises_error(self):
        """Test that unknown model name raises error."""
        with pytest.raises(ValueError, match="Unknown transaction cost model"):
            TransactionCostRegistry.get_transaction_model('nonexistent')

        with pytest.raises(ValueError, match="Unknown slippage model"):
            TransactionCostRegistry.get_slippage_model('nonexistent')

    def test_list_models(self):
        """Test listing all registered models."""
        models = TransactionCostRegistry.list_models()

        assert 'transaction_models' in models
        assert 'slippage_models' in models
        assert 'standard' in models['transaction_models']
        assert 'low_slippage' in models['slippage_models']

    def test_create_combined_from_registry(self, sample_trades):
        """Test creating combined model from registry."""
        combined = TransactionCostRegistry.create_combined_model(
            'standard',
            'low_slippage',
            name='test_combined'
        )

        assert isinstance(combined, CombinedCostModel)
        assert combined.name == 'test_combined'

        costs = combined.calculate_total_cost(sample_trades)
        assert costs['total_cost'] > 0.0

    def test_register_custom_model(self, sample_trades):
        """Test registering a custom model."""
        custom_model = ProportionalCostModel(0.002, name='my_custom')
        TransactionCostRegistry.register_transaction_model(custom_model)

        retrieved = TransactionCostRegistry.get_transaction_model('my_custom')
        assert retrieved.cost_rate == 0.002

        # Clean up
        del TransactionCostRegistry._models['my_custom']


# ============================================================================
# TradeInfo Tests
# ============================================================================

class TestTradeInfo:
    """Tests for TradeInfo dataclass."""

    def test_creation_minimal(self):
        """Test creating TradeInfo with minimal parameters."""
        trade = TradeInfo('AAPL', 100, 150.0, 100000)

        assert trade.ticker == 'AAPL'
        assert trade.shares_traded == 100
        assert trade.price == 150.0
        assert trade.portfolio_value == 100000
        assert trade.daily_volume is None
        assert trade.bid_ask_spread is None

    def test_creation_full(self):
        """Test creating TradeInfo with all parameters."""
        trade = TradeInfo(
            ticker='MSFT',
            shares_traded=-50,
            price=300.0,
            portfolio_value=200000,
            daily_volume=500000,
            bid_ask_spread=0.01
        )

        assert trade.ticker == 'MSFT'
        assert trade.shares_traded == -50
        assert trade.price == 300.0
        assert trade.portfolio_value == 200000
        assert trade.daily_volume == 500000
        assert trade.bid_ask_spread == 0.01


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for transaction cost models."""

    def test_realistic_trading_scenario(self):
        """Test realistic trading scenario with multiple cost components."""
        # Simulate a day of trading
        trades = [
            TradeInfo('AAPL', 100, 175.0, 100000, daily_volume=50000000, bid_ask_spread=0.01),
            TradeInfo('MSFT', -50, 380.0, 100000, daily_volume=30000000, bid_ask_spread=0.015),
            TradeInfo('GOOGL', 25, 140.0, 100000, daily_volume=20000000, bid_ask_spread=0.02),
        ]

        # Use realistic combined model
        transaction_model = ProportionalCostModel(0.001)  # 0.1% commission
        slippage_model = VolumeBasedSlippageModel(
            base_slippage=0.0001,
            volume_impact=0.1
        )

        combined = CombinedCostModel(transaction_model, slippage_model)
        costs = combined.calculate_total_cost(trades)

        # Total cost should be reasonable (< 1% of trade value)
        total_trade_value = sum(abs(t.shares_traded * t.price) for t in trades)
        cost_percentage = costs['total_cost'] / total_trade_value

        assert 0.0 < cost_percentage < 0.01  # Between 0% and 1%
        assert costs['transaction_cost'] > 0.0
        assert costs['slippage'] > 0.0

    def test_comparison_of_models(self, sample_trades):
        """Test that different models produce different costs."""
        models = [
            ProportionalCostModel(0.001, 'low'),
            ProportionalCostModel(0.005, 'high'),
            TieredCostModel(name='tiered'),
            NonlinearCostModel(name='nonlinear')
        ]

        costs = [model.calculate_cost(sample_trades) for model in models]

        # High proportional should cost more than low
        assert costs[1] > costs[0]

        # All should be positive
        assert all(c > 0 for c in costs)

        # All should be different
        assert len(set(costs)) == len(costs)
