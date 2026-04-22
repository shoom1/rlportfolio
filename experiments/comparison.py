"""
Tools for comparing and analyzing multiple experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Any
from .portfolio_tracker import PortfolioExperimentTracker


class ExperimentComparison:
    """Compare multiple portfolio optimization experiments."""

    def __init__(self, tracker: Optional[PortfolioExperimentTracker] = None):
        """
        Initialize comparison tool.

        Args:
            tracker: Portfolio experiment tracker instance
        """
        self.tracker = tracker or PortfolioExperimentTracker()

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple experiments across metrics.

        Args:
            experiment_ids: List of experiment IDs
            metrics: Metrics to compare (None = all)

        Returns:
            Comparison DataFrame
        """
        if metrics is None:
            metrics = [
                'total_return', 'sharpe_ratio', 'sortino_ratio',
                'max_drawdown', 'calmar_ratio', 'win_rate', 'profit_factor'
            ]

        comparison_data = []

        for exp_id in experiment_ids:
            try:
                exp = self.tracker.load_experiment(exp_id)

                row = {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'algorithm': exp.model_info.get('algorithm'),
                    'created_at': exp.created_at
                }

                # Add portfolio metrics
                for metric in metrics:
                    row[metric] = exp.portfolio_metrics.get(metric)

                comparison_data.append(row)

            except Exception as e:
                print(f"Warning: Could not load experiment {exp_id}: {e}")

        df = pd.DataFrame(comparison_data)
        return df

    def plot_comparison(
        self,
        experiment_ids: List[str],
        metric: str = 'sharpe_ratio',
        figsize: tuple = (12, 6)
    ):
        """
        Plot comparison of experiments by metric.

        Args:
            experiment_ids: List of experiment IDs
            metric: Metric to compare
            figsize: Figure size
        """
        df = self.compare_experiments(experiment_ids, metrics=[metric])

        fig, ax = plt.subplots(figsize=figsize)

        # Create bar plot
        bars = ax.bar(range(len(df)), df[metric])

        # Color bars by performance
        colors = plt.cm.RdYlGn(
            (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
        )
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Experiment Comparison: {metric.replace("_", " ").title()}')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['name'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_portfolio_values(
        self,
        experiment_ids: List[str],
        figsize: tuple = (14, 7)
    ):
        """
        Plot portfolio value evolution for multiple experiments.

        Args:
            experiment_ids: List of experiment IDs
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        for exp_id in experiment_ids:
            try:
                exp = self.tracker.load_experiment(exp_id)

                if exp.portfolio_history is not None:
                    label = f"{exp.name} ({exp.model_info.get('algorithm', 'Unknown')})"
                    ax.plot(exp.portfolio_history['value'], label=label, linewidth=2, alpha=0.8)

            except Exception as e:
                print(f"Warning: Could not load experiment {exp_id}: {e}")

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Portfolio Value Evolution Comparison')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_metrics_heatmap(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None,
        figsize: tuple = (12, 8)
    ):
        """
        Create heatmap of normalized metrics across experiments.

        Args:
            experiment_ids: List of experiment IDs
            metrics: Metrics to include
            figsize: Figure size
        """
        if metrics is None:
            metrics = [
                'total_return', 'sharpe_ratio', 'sortino_ratio',
                'calmar_ratio', 'win_rate', 'profit_factor'
            ]

        df = self.compare_experiments(experiment_ids, metrics=metrics)

        # Normalize metrics to 0-1 scale
        normalized = df[metrics].copy()
        for col in metrics:
            if df[col].notna().any():
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    normalized[col] = (df[col] - min_val) / (max_val - min_val)

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            normalized.T,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=df['name'],
            yticklabels=[m.replace('_', ' ').title() for m in metrics],
            cbar_kws={'label': 'Normalized Score'},
            ax=ax
        )

        ax.set_title('Experiment Metrics Comparison (Normalized)')
        plt.tight_layout()
        return fig

    def rank_experiments(
        self,
        experiment_ids: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Rank experiments using weighted scoring.

        Args:
            experiment_ids: List of experiment IDs
            weights: Metric weights (default: equal weighting)

        Returns:
            Ranked DataFrame with scores
        """
        if weights is None:
            weights = {
                'total_return': 0.2,
                'sharpe_ratio': 0.3,
                'sortino_ratio': 0.2,
                'calmar_ratio': 0.15,
                'win_rate': 0.1,
                'profit_factor': 0.05
            }

        df = self.compare_experiments(experiment_ids, metrics=list(weights.keys()))

        # Normalize each metric to 0-1
        normalized_scores = {}
        for metric in weights.keys():
            if df[metric].notna().any():
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:
                    # Invert max_drawdown (lower is better)
                    if metric == 'max_drawdown':
                        normalized_scores[metric] = 1 - ((df[metric] - min_val) / (max_val - min_val))
                    else:
                        normalized_scores[metric] = (df[metric] - min_val) / (max_val - min_val)
                else:
                    normalized_scores[metric] = 0.5  # All same
            else:
                normalized_scores[metric] = 0.0

        # Calculate weighted score
        total_score = np.zeros(len(df))
        for metric, weight in weights.items():
            if metric in normalized_scores:
                total_score += normalized_scores[metric] * weight

        df['score'] = total_score
        df = df.sort_values('score', ascending=False)

        return df

    def generate_report(
        self,
        experiment_ids: List[str],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive comparison report.

        Args:
            experiment_ids: List of experiment IDs
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        df = self.compare_experiments(experiment_ids)
        ranked = self.rank_experiments(experiment_ids)

        report = []
        report.append("=" * 80)
        report.append("EXPERIMENT COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"\nTotal Experiments: {len(experiment_ids)}\n")

        report.append("\n" + "=" * 80)
        report.append("RANKINGS")
        report.append("=" * 80)
        for idx, row in ranked.iterrows():
            report.append(f"\n{ranked.index.get_loc(idx) + 1}. {row['name']}")
            report.append(f"   Algorithm: {row['algorithm']}")
            report.append(f"   Score: {row['score']:.4f}")
            report.append(f"   Sharpe: {row['sharpe_ratio']:.4f} | Return: {row['total_return']:.2%}")

        report.append("\n\n" + "=" * 80)
        report.append("DETAILED METRICS")
        report.append("=" * 80)

        # Format detailed table
        report.append("\n" + df.to_string(index=False))

        report.append("\n\n" + "=" * 80)
        report.append("BEST PERFORMERS BY METRIC")
        report.append("=" * 80)

        metrics = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
        for metric in metrics:
            best_idx = df[metric].idxmax()
            if pd.notna(best_idx):
                report.append(f"\n{metric.replace('_', ' ').title()}: {df.loc[best_idx, 'name']}")
                report.append(f"  Value: {df.loc[best_idx, metric]:.4f}")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")

        return report_text

    def export_comparison(
        self,
        experiment_ids: List[str],
        output_dir: str
    ):
        """
        Export comparison with charts to directory.

        Args:
            experiment_ids: List of experiment IDs
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save comparison table
        df = self.compare_experiments(experiment_ids)
        df.to_csv(output_path / 'comparison.csv', index=False)

        # Save rankings
        ranked = self.rank_experiments(experiment_ids)
        ranked.to_csv(output_path / 'rankings.csv', index=False)

        # Generate and save charts
        fig = self.plot_comparison(experiment_ids, metric='sharpe_ratio')
        fig.savefig(output_path / 'sharpe_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig = self.plot_portfolio_values(experiment_ids)
        fig.savefig(output_path / 'portfolio_values.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig = self.plot_metrics_heatmap(experiment_ids)
        fig.savefig(output_path / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Save report
        report = self.generate_report(experiment_ids)
        with open(output_path / 'report.txt', 'w') as f:
            f.write(report)

        print(f"Comparison exported to {output_dir}")
