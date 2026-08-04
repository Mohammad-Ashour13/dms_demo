from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the final DMS LightGBM model")
    parser.add_argument("--train-windows", type=Path, required=True)
    parser.add_argument("--test-windows", type=Path, required=True)
    parser.add_argument("--feature-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--feature-counts", nargs="+", type=int, default=[7, 20, 40, 65])
    parser.add_argument("--gpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        train_windows_path=args.train_windows,
        test_windows_path=args.test_windows,
        feature_decisions_path=args.feature_decisions,
        output_path=args.output,
        run_mode=args.run_mode,
        seeds=args.seeds,
        feature_counts=args.feature_counts,
        use_gpu=args.gpu,
    )
    print(run_experiment(config))


if __name__ == "__main__":
    main()
