from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from curvature_bench.config import load_experiment_config
from curvature_bench.engine import train_run
from curvature_bench.plots import plot_history


CONFIGS = [
    "configs/experiments/sgd_momentum_mlp.yaml",
    "configs/experiments/adam_mlp.yaml",
    "configs/experiments/adamw_mlp.yaml",
    "configs/experiments/lbfgs_mlp.yaml",
    "configs/experiments/bfgs_tiny_mlp.yaml",
    "configs/experiments/newton_tiny_mlp.yaml",
]


def main() -> None:
    summaries = []

    for config_path in CONFIGS:
        print(f"\n=== Running {config_path} ===")

        cfg = load_experiment_config(config_path)

        try:
            step_df, epoch_df, summary = train_run(cfg)

            out_dir = Path(cfg["output_dir"])
            plot_history(out_dir)

            summary["config_path"] = config_path
            summaries.append(summary)

            print(json.dumps(summary, indent=2))

        except RuntimeError as e:
            print(f"FAILED: {config_path}")
            print(e)

            summaries.append(
                {
                    "config_path": config_path,
                    "failed": True,
                    "error": str(e),
                }
            )

    summary_df = pd.DataFrame(summaries)
    Path("results").mkdir(exist_ok=True)
    summary_df.to_csv("results/mlp_testbed_summary.csv", index=False)

    print("\n=== Summary ===")
    print(summary_df)


if __name__ == "__main__":
    main()