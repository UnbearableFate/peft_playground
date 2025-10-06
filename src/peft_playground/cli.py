"""Command line interface for driving PEFT experiments."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import string
import time
from pathlib import Path

from .config import TrainingConfig
from .training.runners.accelerate_runner import run_accelerated_training
from .training.runners.ddp_runner import run_ddp_training
from .training.runners.trainer_runner import run_trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PEFT playground training CLI")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Instantiate components without running train/eval",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and run evaluation only",
    )
    parser.add_argument(
        "--backend",
        choices=["trainer", "accelerate", "ddp"],
        default="trainer",
        help="Select Hugging Face Trainer, Accelerate, or torch.distributed backend",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Override path for run summary JSON",
    )
    parser.add_argument(
        "--timestamp",
        default=time.strftime("%Y%m%d_%H%M%S"),
        help="Optional explicit timestamp suffix for the output directory",
    )
    return parser.parse_args()

def _write_summary(metrics: dict, path: Path) -> None:
    if not metrics:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Metrics written to {path}")


def main() -> None:
    args = _parse_args()
    cfg = TrainingConfig.load(args.config, args.timestamp)
    summary_path = Path(args.summary_path) if args.summary_path else Path(cfg.train.output_dir) / "run_summary.json"

    metrics: dict | None = None
    wandb_run = None
    
    
    try:
        if args.backend == "trainer":
            metrics, wandb_run = run_trainer(
                cfg,
                evaluate_only=args.evaluate_only,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                trainer_args = metrics.get("trainer_args", {}) if isinstance(metrics, dict) else {}
                print("Dry run: trainer prepared with the following arguments:")
                print(json.dumps(trainer_args, indent=2, sort_keys=True))
                return
        elif args.backend == "accelerate":
            metrics, wandb_run = run_accelerated_training(
                cfg,
                evaluate_only=args.evaluate_only,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                print("Dry run completed for Accelerate backend.")
                return
            _write_summary(metrics, summary_path)
        else:
            metrics, wandb_run = run_ddp_training(
                cfg,
                evaluate_only=args.evaluate_only,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                print("Dry run completed for DDP backend.")
                return
            if dist_metrics := metrics:
                _write_summary(dist_metrics, summary_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    if metrics and args.backend == "trainer" and not args.dry_run:
        _write_summary(metrics, summary_path)


if __name__ == "__main__":
    main()
