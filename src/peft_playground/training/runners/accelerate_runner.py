"""Accelerate-based training runner."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import torch
from accelerate import Accelerator

from ...config import TrainingConfig
from .base_runner import BaseRunner


class AccelerateRunner(BaseRunner):
    """Runner for training with Hugging Face Accelerate."""

    def __init__(
        self,
        cfg: TrainingConfig,
        evaluate_only: bool = False,
        dry_run: bool = False,
    ):
        super().__init__(cfg, evaluate_only, dry_run)
        self.accelerator: Accelerator = None

    def setup(self) -> None:
        grad_accum = self.cfg.accelerate.gradient_accumulation_steps or self.train_cfg.gradient_accumulation_steps
        grad_accum = max(int(grad_accum or 1), 1)

        log_with = self.cfg.accelerate.log_with if self.cfg.accelerate.log_with else None

        self.accelerator = Accelerator(
            gradient_accumulation_steps=grad_accum,
            mixed_precision=self.cfg.accelerate.mixed_precision,
            log_with=log_with,
            project_dir=self.cfg.accelerate.project_dir,
            split_batches=self.cfg.accelerate.split_batches,
            even_batches=self.cfg.accelerate.even_batches,
        )
        self.grad_accum = self.accelerator.gradient_accumulation_steps

    def prepare(self) -> None:
        self.state.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.state.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        )

    def backward(self, loss: torch.Tensor) -> None:
        self.accelerator.backward(loss)

    def get_step_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return self.accelerator.reduce(loss.detach(), reduction="mean")

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.accelerator.gather_for_metrics(tensor)

    @property
    def is_main_process(self) -> bool:
        return self.accelerator.is_main_process

    def wait_for_everyone(self) -> None:
        self.accelerator.wait_for_everyone()

    def unwrap_model(self) -> torch.nn.Module:
        return self.accelerator.unwrap_model(self.state.model)

    def log(self, data: Dict[str, Any], step: int) -> None:
        if self.is_main_process:
            self.accelerator.log(data, step=step)
            if self.state.wandb_run is not None:
                self.state.wandb_run.log(data, step=step)

    def save_checkpoint(self, output_dir: Path) -> None:
        if self.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.accelerator.save_state(output_dir)
            self.state.tokenizer.save_pretrained(output_dir)
            print(f"Checkpoint saved to {output_dir}")

    def load_checkpoint(self, resume_from: Path) -> None:
        self.accelerator.load_state(resume_from)
        self.completed_steps = self.lr_scheduler.last_epoch
        self.starting_epoch = self.completed_steps // (math.ceil(len(self.train_dataloader) / self.grad_accum))
        print(f"Resumed from checkpoint: {resume_from}. Starting at epoch {self.starting_epoch}, step {self.completed_steps}")


def run_accelerated_training(
    cfg: TrainingConfig,
    *,
    evaluate_only: bool = False,
    dry_run: bool = False,
) -> tuple[Dict[str, Any], Any]:
    runner = AccelerateRunner(cfg, evaluate_only=evaluate_only, dry_run=dry_run)
    return runner.run()
