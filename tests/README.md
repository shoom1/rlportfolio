# Unit Tests for RL Portfolio Optimization

This directory contains comprehensive unit tests for the RL portfolio optimization system.

## Test Coverage

The test suite includes **164 tests** covering all major components:

### Data Module Tests (test_fetcher.py, test_features.py)
- **test_fetcher.py**: Tests for data fetching and caching
  - Cache management
  - yfinance integration
  - Data validation
  - Error handling

- **test_features.py**: Tests for feature engineering
  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
  - Returns and volatility calculations
  - Feature registry and composition
  - Data normalization and preparation

### Environment Module Tests (test_portfolio_env.py, test_rewards.py)
- **test_portfolio_env.py**: Tests for the Gymnasium environment
  - Environment initialization and configuration
  - Action and observation spaces
  - Portfolio rebalancing logic
  - Transaction cost calculations
  - Episode management
  - Data validation

- **test_rewards.py**: Tests for reward functions
  - Simple return and log return rewards
  - Sharpe and Sortino ratio rewards
  - Risk-adjusted return rewards
  - Drawdown-penalized rewards
  - Reward function factory

### Evaluation Module Tests (test_metrics.py, test_baselines.py, test_backtest.py)
- **test_metrics.py**: Tests for performance metrics
  - Return calculations (total, annualized)
  - Risk metrics (Sharpe, Sortino, Calmar ratios)
  - Drawdown analysis
  - Win/loss statistics
  - Baseline comparison metrics (alpha, beta, information ratio)

- **test_baselines.py**: Tests for baseline strategies
  - Equal weight strategy
  - Buy and hold strategy
  - Random strategy
  - Momentum strategy
  - Minimum variance strategy
  - Inverse volatility strategy
  - Strategy registry

- **test_backtest.py**: Tests for backtesting framework
  - Agent backtesting
  - Baseline strategy backtesting
  - Result storage and retrieval
  - Metrics computation
  - Result comparison and export

## Running the Tests

### Using conda (Recommended)
```bash
# Run all tests
conda run -n rlportfolio python -m pytest tests/

# Run with verbose output
conda run -n rlportfolio python -m pytest tests/ -v

# Run specific test file
conda run -n rlportfolio python -m pytest tests/test_portfolio_env.py -v

# Run tests with coverage report
conda run -n rlportfolio python -m pytest tests/ --cov=. --cov-report=html
```

### Using pip environment
```bash
# Activate your environment first
source venv/bin/activate  # or appropriate activation command

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py -v
```

## Test Configuration

Tests are configured via `pytest.ini` in the project root:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --tb=short
    --disable-warnings
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

## Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_ohlcv_data`: Sample OHLCV market data for 3 tickers
- `sample_portfolio_history`: Sample portfolio history for metrics testing
- `sample_tickers`: List of sample ticker symbols
- `feature_columns`: Standard feature column names

## Test Structure

Each test file follows a consistent structure:

```python
class TestClassName:
    """Tests for ClassName."""

    @pytest.fixture
    def fixture_name(self):
        """Fixture description."""
        # Setup code
        return object

    def test_feature_name(self, fixture_name):
        """Test description."""
        # Arrange
        # Act
        # Assert
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. All tests use:

- Mock objects for external dependencies (yfinance API calls)
- Deterministic random seeds for reproducibility
- Temporary directories for file operations
- Fast execution (< 15 seconds total)

## Writing New Tests

When adding new features, follow these guidelines:

1. **Test Coverage**: Aim for >80% code coverage
2. **Test Independence**: Each test should be independent and idempotent
3. **Mock External Calls**: Use `unittest.mock` for API calls
4. **Use Fixtures**: Share common setup code via pytest fixtures
5. **Descriptive Names**: Use descriptive test names that explain what is being tested
6. **Arrange-Act-Assert**: Follow the AAA pattern for clarity

### Example Test

```python
def test_feature_computes_correctly(self, sample_data):
    """Test that feature is computed with correct values."""
    # Arrange
    feature = MyFeature(param=10)

    # Act
    result = feature.compute(sample_data)

    # Assert
    assert 'my_feature' in result.columns
    assert not result['my_feature'].isna().any()
```

## Current Test Results

✅ **164 tests passing**
- 25 tests for data fetching and features
- 54 tests for environment and rewards
- 60 tests for metrics and evaluation
- 25 tests for baselines and backtesting

## Dependencies

Tests require:
- pytest >= 7.4.0
- All project dependencies (see requirements.txt)
- Conda environment recommended for consistency
