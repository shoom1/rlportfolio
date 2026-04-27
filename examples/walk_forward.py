"""Example: quarterly walk-forward on the div19 universe, 2005-2026.

Thin wrapper over `evaluation.walk_forward.WalkForwardEvaluator`. For other
universes / algorithms / fold geometries, write a similar driver or use the
CLI::

    python -m evaluation.walk_forward --config configs/opt_c_div19.yaml \
        --seeds 42 43 44
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import TrainingConfig
from evaluation.walk_forward import WalkForwardConfig, WalkForwardEvaluator


CFG_PATH = PROJECT_ROOT / "configs" / "opt_c_div19.yaml"
RESULTS_CSV = PROJECT_ROOT / "results" / "walk_forward_div19.csv"


def main() -> None:
    cfg = TrainingConfig.from_yaml(CFG_PATH)
    wf = WalkForwardConfig(
        seeds=[42, 43, 44],
    )  # quarterly defaults: 3y min train, Q stride, Q tau
    evaluator = WalkForwardEvaluator(cfg=cfg, wf=wf)

    df = evaluator.run()

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults written to {RESULTS_CSV}")

    stats = WalkForwardEvaluator.summarise(df, wf.baselines)
    print("\n" + "=" * 72)
    print(
        f"AGGREGATE over {stats['n_successful_runs']}/{stats['n_runs']} "
        f"successful runs across {stats['n_folds']} folds"
    )
    print("=" * 72)
    a = stats['agent_sharpe']
    print(f"agent Sharpe: mean={a['mean']:+.3f}  std={a['std']:.3f}  "
          f"median={a['median']:+.3f}")
    af = stats['agent_sharpe_by_fold']
    print(f"agent fold-mean Sharpe: mean={af['mean']:+.3f}  std={af['std']:.3f}")
    for b in wf.baselines:
        s = stats[f'{b}_sharpe']
        print(f"{b:>20s} Sharpe: mean={s['mean']:+.3f}  median={s['median']:+.3f}   "
              f"agent > {b}: {stats[f'{b}_hit_rate']:.1%} of successful runs")


if __name__ == "__main__":
    main()
