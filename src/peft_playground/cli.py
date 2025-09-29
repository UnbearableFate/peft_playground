"""Command line interface for driving PEFT experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import TrainingConfig
from .pipeline import prepare_training_components


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
    return parser.parse_args()


def _maybe_run_training(trainer, evaluate_only: bool) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not evaluate_only and trainer.args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        trainer.save_state()
        metrics.update({f"train_{k}": v for k, v in train_result.metrics.items()})
    if trainer.args.do_eval or evaluate_only:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
    if trainer.args.push_to_hub:
        trainer.push_to_hub()
    return metrics


def main() -> None:
    args = _parse_args()
    cfg = TrainingConfig.load(args.config)
    components = prepare_training_components(cfg)
    trainer = components["trainer"]
    wandb_run = components.get("wandb_run")

    try:
        if args.dry_run:
            print("Dry run: trainer prepared with the following arguments:")
            print(json.dumps(trainer.args.to_dict(), indent=2, sort_keys=True))
            return

        metrics = _maybe_run_training(trainer, args.evaluate_only)
        if metrics:
            summary_path = Path(trainer.args.output_dir) / "run_summary.json"
            summary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
            print(f"Metrics written to {summary_path}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

