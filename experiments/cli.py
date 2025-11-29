"""
Command-line interface for experiment management.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from .tracker import PortfolioExperimentTracker
from .comparison import ExperimentComparison


console = Console()


def list_experiments(args):
    """List all experiments."""
    tracker = PortfolioExperimentTracker(storage_dir=args.storage_dir)

    # Query experiments
    df = tracker.query_experiments(
        tags=args.tags,
        algorithm=args.algorithm,
        min_sharpe=args.min_sharpe,
        min_return=args.min_return,
        limit=args.limit
    )

    if df.empty:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    # Create rich table
    table = Table(title="Portfolio Experiments")

    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Algorithm", style="green")
    table.add_column("Total Return", justify="right")
    table.add_column("Sharpe Ratio", justify="right")
    table.add_column("Max Drawdown", justify="right")
    table.add_column("Created", style="dim")

    for _, row in df.iterrows():
        table.add_row(
            row['experiment_id'][:12] + "...",
            row['name'],
            row['algorithm'] or "N/A",
            f"{row['total_return']:.2%}" if row['total_return'] else "N/A",
            f"{row['sharpe_ratio']:.4f}" if row['sharpe_ratio'] else "N/A",
            f"{row['max_drawdown']:.2%}" if row['max_drawdown'] else "N/A",
            row['created_at'][:10] if row['created_at'] else "N/A"
        )

    console.print(table)
    console.print(f"\n[bold]Total experiments:[/bold] {len(df)}")


def show_experiment(args):
    """Show detailed experiment information."""
    tracker = PortfolioExperimentTracker(storage_dir=args.storage_dir)

    try:
        exp = tracker.load_experiment(args.experiment_id)

        console.print("\n" + "="*80)
        console.print(f"[bold cyan]Experiment: {exp.name}[/bold cyan]")
        console.print("="*80)

        console.print(f"\n[bold]ID:[/bold] {exp.experiment_id}")
        console.print(f"[bold]Description:[/bold] {exp.description}")
        console.print(f"[bold]Tags:[/bold] {', '.join(exp.tags)}")
        console.print(f"[bold]Created:[/bold] {exp.created_at}")

        if exp.wandb_run_id:
            console.print(f"[bold]W&B Run ID:[/bold] {exp.wandb_run_id}")

        # Portfolio configuration
        console.print("\n[bold yellow]Portfolio Configuration:[/bold yellow]")
        for key, value in exp.portfolio_config.items():
            console.print(f"  {key}: {value}")

        # Model info
        console.print("\n[bold green]Model Information:[/bold green]")
        console.print(f"  Algorithm: {exp.model_info.get('algorithm')}")
        console.print(f"  Policy: {exp.model_info.get('policy')}")
        console.print(f"  Architecture: {exp.model_info.get('policy_architecture')}")

        # Metrics
        console.print("\n[bold blue]Performance Metrics:[/bold blue]")
        metrics_table = Table()
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", justify="right", style="green")

        for metric, value in exp.portfolio_metrics.items():
            if value is not None:
                if isinstance(value, float):
                    if 'return' in metric or 'drawdown' in metric:
                        formatted = f"{value:.2%}"
                    else:
                        formatted = f"{value:.4f}"
                else:
                    formatted = str(value)

                metrics_table.add_row(metric.replace('_', ' ').title(), formatted)

        console.print(metrics_table)

        # Baseline comparison
        if exp.baseline_comparison:
            console.print("\n[bold magenta]Baseline Comparison:[/bold magenta]")
            comp_table = Table()
            comp_table.add_column("Strategy", style="cyan")
            comp_table.add_column("Total Return", justify="right")
            comp_table.add_column("Sharpe Ratio", justify="right")

            for baseline, metrics in exp.baseline_comparison.items():
                comp_table.add_row(
                    baseline,
                    f"{metrics.get('total_return', 0):.2%}",
                    f"{metrics.get('sharpe_ratio', 0):.4f}"
                )

            console.print(comp_table)

        console.print("\n" + "="*80 + "\n")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


def compare_experiments(args):
    """Compare multiple experiments."""
    tracker = PortfolioExperimentTracker(storage_dir=args.storage_dir)
    comparison = ExperimentComparison(tracker)

    try:
        # Generate comparison report
        report = comparison.generate_report(
            args.experiment_ids,
            output_path=args.output if args.output else None
        )

        console.print(report)

        # Export if requested
        if args.export:
            console.print(f"\n[green]Exporting comparison to {args.export}...[/green]")
            comparison.export_comparison(args.experiment_ids, args.export)
            console.print("[green]Export complete![/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def rank_experiments(args):
    """Rank experiments by performance."""
    tracker = PortfolioExperimentTracker(storage_dir=args.storage_dir)
    comparison = ExperimentComparison(tracker)

    # Get all experiments if not specified
    if not args.experiment_ids:
        df = tracker.query_experiments(limit=100)
        experiment_ids = df['experiment_id'].tolist()
    else:
        experiment_ids = args.experiment_ids

    # Rank experiments
    ranked = comparison.rank_experiments(experiment_ids)

    # Display rankings
    table = Table(title="Experiment Rankings")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Score", justify="right", style="bold green")
    table.add_column("Sharpe", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Algorithm", style="dim")

    for idx, row in ranked.head(args.top_n).iterrows():
        rank = ranked.index.get_loc(idx) + 1
        table.add_row(
            str(rank),
            row['name'],
            f"{row['score']:.4f}",
            f"{row['sharpe_ratio']:.4f}",
            f"{row['total_return']:.2%}",
            row['algorithm'] or "N/A"
        )

    console.print(table)


def delete_experiment(args):
    """Delete an experiment."""
    tracker = PortfolioExperimentTracker(storage_dir=args.storage_dir)

    if not args.force:
        response = input(f"Delete experiment {args.experiment_id}? (y/N): ")
        if response.lower() != 'y':
            console.print("[yellow]Cancelled.[/yellow]")
            return

    try:
        tracker.delete_experiment(args.experiment_id)
        console.print(f"[green]Deleted experiment {args.experiment_id}[/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


def best_experiments(args):
    """Show best experiments by metric."""
    tracker = PortfolioExperimentTracker(storage_dir=args.storage_dir)

    df = tracker.get_best_experiments(metric=args.metric, top_n=args.top_n)

    if df.empty:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    table = Table(title=f"Top {args.top_n} Experiments by {args.metric.replace('_', ' ').title()}")

    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column(args.metric.replace('_', ' ').title(), justify="right", style="bold green")
    table.add_column("Algorithm", style="dim")
    table.add_column("Created", style="dim")

    for idx, row in df.iterrows():
        rank = df.index.get_loc(idx) + 1
        metric_value = row[args.metric]

        if metric_value:
            if 'return' in args.metric or 'drawdown' in args.metric:
                metric_str = f"{metric_value:.2%}"
            else:
                metric_str = f"{metric_value:.4f}"
        else:
            metric_str = "N/A"

        table.add_row(
            str(rank),
            row['name'],
            metric_str,
            row['algorithm'] or "N/A",
            row['created_at'][:10] if row['created_at'] else "N/A"
        )

    console.print(table)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment Management CLI for RL Portfolio Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--storage-dir',
        type=str,
        default='experiments/portfolio',
        help='Experiments storage directory'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--tags', nargs='+', help='Filter by tags')
    list_parser.add_argument('--algorithm', type=str, help='Filter by algorithm')
    list_parser.add_argument('--min-sharpe', type=float, help='Minimum Sharpe ratio')
    list_parser.add_argument('--min-return', type=float, help='Minimum return')
    list_parser.add_argument('--limit', type=int, default=50, help='Max results')
    list_parser.set_defaults(func=list_experiments)

    # Show command
    show_parser = subparsers.add_parser('show', help='Show experiment details')
    show_parser.add_argument('experiment_id', help='Experiment ID')
    show_parser.set_defaults(func=show_experiment)

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('experiment_ids', nargs='+', help='Experiment IDs')
    compare_parser.add_argument('--output', type=str, help='Save report to file')
    compare_parser.add_argument('--export', type=str, help='Export to directory')
    compare_parser.set_defaults(func=compare_experiments)

    # Rank command
    rank_parser = subparsers.add_parser('rank', help='Rank experiments')
    rank_parser.add_argument('experiment_ids', nargs='*', help='Experiment IDs (empty = all)')
    rank_parser.add_argument('--top-n', type=int, default=10, help='Show top N')
    rank_parser.set_defaults(func=rank_experiments)

    # Best command
    best_parser = subparsers.add_parser('best', help='Show best experiments')
    best_parser.add_argument('--metric', type=str, default='sharpe_ratio',
                           help='Metric to rank by')
    best_parser.add_argument('--top-n', type=int, default=10, help='Show top N')
    best_parser.set_defaults(func=best_experiments)

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete experiment')
    delete_parser.add_argument('experiment_id', help='Experiment ID')
    delete_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    delete_parser.set_defaults(func=delete_experiment)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == '__main__':
    main()
