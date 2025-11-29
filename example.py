"""
Simple example script demonstrating the complete workflow.
Run this to verify installation and see the framework in action.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from data.fetcher import DataFetcher
from data.features import FeatureEngineer
from environment import PortfolioEnv, SharpeReward
from evaluation import Backtester, plot_strategy_comparison, plot_portfolio_weights
from stable_baselines3 import PPO


def main():
    print("="*80)
    print("RL Portfolio Optimization - Example Workflow")
    print("="*80)

    # 1. Fetch Data
    print("\n[1/6] Fetching market data...")
    fetcher = DataFetcher()
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = fetcher.get_latest_data(tickers, days=365)
    print(f"✓ Fetched {len(data)} data points for {len(tickers)} tickers")

    # 2. Engineer Features
    print("\n[2/6] Computing technical indicators...")
    engineer = FeatureEngineer()
    features_data = engineer.compute_features(data)
    env_data = engineer.prepare_for_environment(features_data)
    print(f"✓ Computed {len(features_data.columns)} features")

    # 3. Create Environment
    print("\n[3/6] Creating portfolio environment...")
    feature_cols = engineer.create_observation_columns(normalize=True)
    env = PortfolioEnv(
        data=env_data,
        feature_columns=feature_cols,
        tickers=tickers,
        initial_balance=10000.0,
        transaction_cost=0.001,
        reward_function=SharpeReward(window=30),
        window_size=20
    )
    print(f"✓ Environment created")
    print(f"  - Observation space: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space.shape}")
    print(f"  - Episode length: {env.n_steps} steps")

    # 4. Train Agent (quick demo)
    print("\n[4/6] Training PPO agent (quick demo - 5000 steps)...")
    print("  Note: For real training, use agents/train.py with higher timesteps")
    agent = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        policy_kwargs=dict(net_arch=[128, 128])
    )
    agent.learn(total_timesteps=5000, progress_bar=True)
    print("✓ Training completed")

    # 5. Run Backtests
    print("\n[5/6] Running backtests...")
    backtester = Backtester()

    # Create fresh environments for each backtest
    def create_env():
        return PortfolioEnv(
            data=env_data,
            feature_columns=feature_cols,
            tickers=tickers,
            initial_balance=10000.0,
            transaction_cost=0.001,
            reward_function=SharpeReward(window=30),
            window_size=20
        )

    backtester.run_agent(agent, create_env(), name='PPO Agent')
    backtester.run_baseline(create_env(), 'equal_weight')
    backtester.run_baseline(create_env(), 'momentum_20')
    backtester.run_baseline(create_env(), 'buy_and_hold')
    print("✓ Backtests completed")

    # 6. Analyze Results
    print("\n[6/6] Analyzing results...")
    backtester.print_comparison()

    # Get histories for visualization
    histories = backtester.get_histories()

    print("\n" + "="*80)
    print("Example completed successfully!")
    print("\nNext steps:")
    print("1. Train for longer: python agents/train.py")
    print("2. Try different configs: python agents/train.py --config configs/sac_config.yaml")
    print("3. Explore the code in notebooks/")
    print("="*80)

    # Optional: Show plots if matplotlib available
    try:
        plot_strategy_comparison(histories)
        result = backtester.get_result('PPO Agent')
        plot_portfolio_weights(result.history, tickers)
    except Exception as e:
        print(f"\nNote: Could not display plots: {e}")
        print("Plots can be saved with save_path parameter")


if __name__ == "__main__":
    main()
