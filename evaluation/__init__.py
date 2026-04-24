"""
Evaluation and backtesting modules.

Refactored with design patterns:
- Simple Composition for metrics calculation
- Strategy pattern for backtest execution
"""

from .metrics import (
    PortfolioMetrics,
    ReturnMetrics,
    RiskMetrics,
    DrawdownMetrics,
    WinLossMetrics,
    TransactionCostMetrics,
    BaselineComparisonMetrics
)
from .backtest import Backtester, BacktestResult, run_agent_backtest, run_baseline_backtest
from .backtest_strategies import (
    BacktestStrategy,
    SequentialBacktestStrategy,
    WalkForwardBacktestStrategy,
    MonteCarloBacktestStrategy
)
from .baselines import (
    BaselineStrategy,
    BaselineStrategyRegistry,
    EqualWeightStrategy,
    BuyAndHoldStrategy,
    RandomStrategy,
    MomentumStrategy,
    MinimumVarianceStrategy,
    InverseVolatilityStrategy
)
from .visualization import (
    plot_strategy_comparison,
    plot_portfolio_weights,
    plot_risk_return_scatter,
    plot_rolling_metrics,
    plot_monthly_returns_heatmap,
    plot_underwater
)
from .visualize_network import (
    print_network_structure,
    print_layer_dimensions,
    create_ascii_diagram,
    analyze_network_from_config
)
# NOTE: evaluation.walk_forward is a submodule, not auto-exported here.
# Import it explicitly:
#     from evaluation.walk_forward import WalkForwardEvaluator, WalkForwardConfig
# Keeping it out of __init__ avoids a runpy double-import warning when the
# module is run via `python -m evaluation.walk_forward`.

__all__ = [
    # Metrics - Composition pattern
    'PortfolioMetrics',
    'ReturnMetrics',
    'RiskMetrics',
    'DrawdownMetrics',
    'WinLossMetrics',
    'TransactionCostMetrics',
    'BaselineComparisonMetrics',
    # Backtesting
    'Backtester',
    'BacktestResult',
    'run_agent_backtest',
    'run_baseline_backtest',
    # Backtest Strategies - Strategy pattern
    'BacktestStrategy',
    'SequentialBacktestStrategy',
    'WalkForwardBacktestStrategy',
    'MonteCarloBacktestStrategy',
    # Baselines
    'BaselineStrategy',
    'BaselineStrategyRegistry',
    'EqualWeightStrategy',
    'BuyAndHoldStrategy',
    'RandomStrategy',
    'MomentumStrategy',
    'MinimumVarianceStrategy',
    'InverseVolatilityStrategy',
    # Visualization
    'plot_strategy_comparison',
    'plot_portfolio_weights',
    'plot_risk_return_scatter',
    'plot_rolling_metrics',
    'plot_monthly_returns_heatmap',
    'plot_underwater',
    # Network visualization
    'print_network_structure',
    'print_layer_dimensions',
    'create_ascii_diagram',
    'analyze_network_from_config',
]
