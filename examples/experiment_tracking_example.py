"""
Example demonstrating experiment tracking capabilities.
Shows how to use W&B + custom tracker for portfolio optimization.
"""

import sys
sys.path.append('..')

from experiments import (
    PortfolioExperimentTracker,
    ExperimentComparison,
    WandbTracker
)


def example_1_basic_tracking():
    """Example 1: Basic experiment tracking without training."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Experiment Tracking")
    print("="*80)

    # Create tracker
    tracker = PortfolioExperimentTracker()

    # Create an experiment
    exp = tracker.create_experiment(
        name='example_experiment',
        description='Demo of portfolio experiment tracking',
        tags=['example', 'demo'],
        config={
            'tickers': ['AAPL', 'MSFT', 'GOOGL'],
            'initial_balance': 10000.0,
            'transaction_cost': 0.001
        }
    )

    print(f"Created experiment: {exp.experiment_id}")

    # Simulate setting model info
    exp.model_info.update({
        'algorithm': 'PPO',
        'policy': 'MlpPolicy',
        'hyperparameters': {
            'learning_rate': 0.0003,
            'batch_size': 64
        }
    })

    # Simulate setting metrics
    exp.set_portfolio_metrics({
        'total_return': 0.25,
        'sharpe_ratio': 1.5,
        'sortino_ratio': 1.8,
        'max_drawdown': -0.12,
        'win_rate': 0.62
    })

    # Add baseline comparison
    exp.add_baseline_comparison('equal_weight', {
        'total_return': 0.15,
        'sharpe_ratio': 1.0
    })

    # Save
    tracker.finish_experiment()

    print(f"Experiment saved with Sharpe ratio: {exp.portfolio_metrics['sharpe_ratio']}")


def example_2_query_experiments():
    """Example 2: Querying experiments."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Querying Experiments")
    print("="*80)

    tracker = PortfolioExperimentTracker()

    # Query all experiments
    all_exp = tracker.query_experiments(limit=10)
    print(f"\nTotal experiments: {len(all_exp)}")

    if not all_exp.empty:
        print("\nRecent experiments:")
        print(all_exp[['name', 'algorithm', 'sharpe_ratio', 'total_return']].head())

    # Query with filters
    good_exp = tracker.query_experiments(
        min_sharpe=1.0,
        min_return=0.1,
        limit=5
    )

    print(f"\nExperiments with Sharpe>1.0 and Return>10%: {len(good_exp)}")

    # Get best by metric
    best = tracker.get_best_experiments(metric='sharpe_ratio', top_n=5)
    print("\nTop 5 by Sharpe ratio:")
    if not best.empty:
        print(best[['name', 'sharpe_ratio', 'total_return']])


def example_3_compare_experiments():
    """Example 3: Comparing experiments."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Comparing Experiments")
    print("="*80)

    tracker = PortfolioExperimentTracker()
    comparison = ExperimentComparison(tracker)

    # Get some experiments to compare
    all_exp = tracker.query_experiments(limit=10)

    if len(all_exp) >= 2:
        exp_ids = all_exp['experiment_id'].head(3).tolist()

        print(f"\nComparing {len(exp_ids)} experiments...")

        # Compare
        df = comparison.compare_experiments(
            exp_ids,
            metrics=['sharpe_ratio', 'total_return', 'max_drawdown']
        )

        print("\nComparison:")
        print(df[['name', 'sharpe_ratio', 'total_return', 'max_drawdown']])

        # Rank
        ranked = comparison.rank_experiments(exp_ids)
        print("\nRankings:")
        print(ranked[['name', 'score', 'sharpe_ratio', 'total_return']])

        # Generate report
        report = comparison.generate_report(exp_ids)
        print("\n" + report[:500] + "...")  # Print first 500 chars

    else:
        print("Not enough experiments for comparison. Create more first!")


def example_4_wandb_integration():
    """Example 4: W&B integration (requires W&B account)."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Weights & Biases Integration")
    print("="*80)

    print("""
This example shows how to use W&B integration with the new Strategy pattern.
To run this, you need:
1. W&B account (free at wandb.ai)
2. Run 'wandb login' to authenticate

Example code:

    from experiments import WandbTracker
    import wandb

    # Initialize W&B tracker
    wandb_tracker = WandbTracker(
        experiment_name='example_run',
        config={
            'algorithm': 'PPO',
            'learning_rate': 0.0003
        }
    )

    # Use context manager for automatic initialization/cleanup
    with wandb_tracker.track(
        project='rl-portfolio-demo',
        tags=['example', 'demo']
    ):
        # During training, log metrics
        wandb_tracker.log_metrics({
            'train/reward': 0.5,
            'portfolio/value': 10500
        }, step=100)

        # Log hyperparameters
        wandb_tracker.log_hyperparameters({
            'batch_size': 64,
            'gamma': 0.99
        })

        # Log artifacts (models, results, etc.)
        # wandb_tracker.log_artifact('model.zip', artifact_type='model')

See experiments/README.md for full W&B integration guide.
    """)


def example_5_full_workflow():
    """Example 5: Complete workflow demonstration."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Complete Workflow")
    print("="*80)

    tracker = PortfolioExperimentTracker()

    # Create multiple experiments
    print("\nCreating 3 demo experiments...")

    configs = [
        {'algorithm': 'PPO', 'lr': 0.0003, 'sharpe': 1.5, 'return': 0.25},
        {'algorithm': 'SAC', 'lr': 0.0001, 'sharpe': 1.8, 'return': 0.30},
        {'algorithm': 'A2C', 'lr': 0.0007, 'sharpe': 1.2, 'return': 0.18},
    ]

    exp_ids = []
    for i, cfg in enumerate(configs):
        exp = tracker.create_experiment(
            name=f"demo_{cfg['algorithm'].lower()}",
            description=f"Demo {cfg['algorithm']} experiment",
            tags=['demo', 'workflow'],
            config={'algorithm': cfg['algorithm'], 'learning_rate': cfg['lr']}
        )

        exp.model_info['algorithm'] = cfg['algorithm']
        exp.set_portfolio_metrics({
            'sharpe_ratio': cfg['sharpe'],
            'total_return': cfg['return'],
            'max_drawdown': -0.10 - i*0.02
        })

        tracker.finish_experiment()
        exp_ids.append(exp.experiment_id)
        print(f"  Created: {exp.name} (Sharpe: {cfg['sharpe']})")

    # Query and compare
    print("\nQuerying experiments...")
    results = tracker.query_experiments(tags=['demo'])
    print(f"Found {len(results)} demo experiments")

    # Compare
    print("\nComparing experiments...")
    comparison = ExperimentComparison(tracker)

    df = comparison.compare_experiments(
        exp_ids,
        metrics=['sharpe_ratio', 'total_return']
    )

    print("\nComparison Results:")
    print(df[['name', 'algorithm', 'sharpe_ratio', 'total_return']])

    # Rank
    ranked = comparison.rank_experiments(exp_ids)
    print("\nRankings:")
    for idx, row in ranked.iterrows():
        rank = ranked.index.get_loc(idx) + 1
        print(f"  {rank}. {row['name']} - Score: {row['score']:.4f}")

    # Best by metric
    best = tracker.get_best_experiments(metric='sharpe_ratio', top_n=1)
    if not best.empty:
        print(f"\nBest experiment: {best.iloc[0]['name']} (Sharpe: {best.iloc[0]['sharpe_ratio']:.4f})")

    print("\n" + "="*80)
    print("Workflow complete! Check experiments/portfolio/ for saved data.")
    print("="*80)


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("PORTFOLIO EXPERIMENT TRACKING EXAMPLES")
    print("="*80)

    # Run examples
    example_1_basic_tracking()
    example_2_query_experiments()
    example_3_compare_experiments()
    example_4_wandb_integration()
    example_5_full_workflow()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Try the CLI: python -m experiments.cli list")
    print("2. Run actual training with tracking:")
    print("   python agents/train_with_tracking.py --config configs/ppo_config.yaml")
    print("3. Set up W&B: wandb login")
    print("4. Read experiments/README.md for full documentation")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
