from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvature_bench.config import load_experiment_config
from curvature_bench.engine import train_run
from curvature_bench.plots import plot_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_experiment_config(args.config)

    step_df, epoch_df, summary = train_run(cfg)

    out_dir = Path(cfg["output_dir"])
    plot_history(out_dir)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()