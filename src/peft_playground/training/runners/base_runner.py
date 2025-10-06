"""Base class for custom training runners."""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from ...config import TrainingConfig
from ..preparation import build_training_state
from ..state import TrainingState


class BaseRunner(ABC):
    """Abstract base class for training runners."""

    def __init__(
        self,
        cfg: TrainingConfig,
        evaluate_only: bool = False,
        dry_run: bool = False,
    ):
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.eval_cfg = cfg.evaluation
        self.evaluate_only = evaluate_only
        self.dry_run = dry_run
        self.state: Optional[TrainingState] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.eval_dataloader: Optional[DataLoader] = None
        self.max_train_steps: int = 0
        self.num_epochs: int = 0
        self.grad_accum: int = 1
        
        self.output_dir = Path(self.train_cfg.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.best_model_dir = self.output_dir / "best_model"
        
        self.metric_for_best_model = self.eval_cfg.metric_for_best_model
        self.metric_greater_is_better = self.eval_cfg.greater_is_better
        self.best_metric_val: float = -1.0 if self.metric_greater_is_better else float("inf")
        
        self.completed_steps = 0
        self.starting_epoch = 0
        self.current_epoch = 0

    @abstractmethod
    def setup(self) -> None:
        """Initialize the distributed environment and devices."""
        pass

    @abstractmethod
    def prepare(self) -> None:
        """Prepare model, optimizer, dataloaders for training."""
        pass

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass."""
        pass

    @abstractmethod
    def get_step_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Get loss for a single step, potentially reduced across processes."""
        pass

    @abstractmethod
    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors for metric computation."""
        pass

    @property
    @abstractmethod
    def is_main_process(self) -> bool:
        """Whether the current process is the main one."""
        pass

    @abstractmethod
    def wait_for_everyone(self) -> None:
        """Wait for all processes to sync."""
        pass

    @abstractmethod
    def unwrap_model(self) -> torch.nn.Module:
        """Unwrap the model from any wrappers."""
        pass

    @abstractmethod
    def log(self, data: Dict[str, Any], step: int) -> None:
        """Log data."""
        pass
        
    @abstractmethod
    def save_checkpoint(self, output_dir: Path) -> None:
        """Save a training checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(self, resume_from: Path) -> None:
        """Load a training checkpoint."""
        pass

    def build_state(self) -> None:
        """Build the initial training state."""
        self.state = build_training_state(self.cfg, init_wandb=self.is_main_process and self.cfg.wandb.enabled)

    def create_dataloaders(self) -> None:
        """Create training and evaluation dataloaders."""
        num_workers = int(self.train_cfg.dataloader_num_workers or 0)
        self.train_dataloader = DataLoader(
            self.state.train_dataset,
            shuffle=True,
            batch_size=self.train_cfg.per_device_batch_size,
            collate_fn=self.state.data_collator,
            num_workers=num_workers,
        )
        self.eval_dataloader = DataLoader(
            self.state.eval_dataset,
            shuffle=False,
            batch_size=self.eval_cfg.per_device_batch_size,
            collate_fn=self.state.data_collator,
            num_workers=num_workers,
        )

    def create_optimizer_and_scheduler(self) -> None:
        """Create optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.AdamW(
            [p for p in self.state.model.parameters() if p.requires_grad],
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )

        self.grad_accum = max(int(self.train_cfg.gradient_accumulation_steps or 1), 1)
        train_len = max(1, len(self.train_dataloader))
        num_update_steps_per_epoch = math.ceil(train_len / self.grad_accum)
        
        if self.train_cfg.max_steps:
            self.max_train_steps = int(self.train_cfg.max_steps)
        else:
            self.max_train_steps = int(math.ceil(self.train_cfg.num_epochs * num_update_steps_per_epoch))
        
        self.num_epochs = max(1, math.ceil(self.max_train_steps / num_update_steps_per_epoch))

        warmup_steps = int(self.train_cfg.warmup_ratio * self.max_train_steps)
        scheduler_name = self.train_cfg.lr_scheduler_type or "linear"
        self.lr_scheduler = get_scheduler(
            scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.max_train_steps,
        )

    def run(self) -> Tuple[Dict[str, Any], Any]:
        """Main entry point to run training and evaluation."""
        self.setup()
        self.build_state()
        self.create_dataloaders()
        self.create_optimizer_and_scheduler()
        self.prepare()
        
        print(f"***** Running training ***** {self.train_cfg}")

        if self.train_cfg.resume_from_checkpoint:
            resume_path = Path(self.train_cfg.resume_from_checkpoint)
            print(f"Resuming from checkpoint: {resume_path}")
            self.load_checkpoint(resume_path)

        if self.dry_run:
            if self.is_main_process:
                print("Dry run successful.")
            return {}, self.state.wandb_run if self.is_main_process else None

        if self.evaluate_only:
            eval_metrics = self.run_evaluation(self.completed_steps)
            return eval_metrics, self.state.wandb_run if self.is_main_process else None

        train_metrics = self.run_training()
        
        self.wait_for_everyone()
        if not self.evaluate_only and self.is_main_process:
            self.save_final_model()
        self.wait_for_everyone()

        return train_metrics, self.state.wandb_run if self.is_main_process else None

    def run_training(self) -> Dict[str, float]:
        progress_bar = tqdm(range(self.max_train_steps), disable=not self.is_main_process)
        progress_bar.set_description("training")
        progress_bar.update(self.completed_steps)

        running_loss = 0.0
        logging_steps = int(self.train_cfg.logging_steps or 0)
        eval_strategy = (self.eval_cfg.strategy or "no").lower()
        eval_steps = self.eval_cfg.steps
        save_steps = self.eval_cfg.save_steps
        last_eval: Dict[str, float] = {}
        start_time = time.time()

        for epoch in range(self.starting_epoch, self.num_epochs):
            self.state.model.train()
            for step, batch in enumerate(self.train_dataloader):
                if self.completed_steps >= self.max_train_steps:
                    break

                outputs = self.state.model(**batch)
                loss = outputs.loss
                self.backward(loss)
                
                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(self.train_dataloader):
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.completed_steps += 1
                    progress_bar.update(1)

                    step_loss = self.get_step_loss(loss).item()
                    running_loss += step_loss

                    if logging_steps and self.completed_steps % logging_steps == 0:
                        logs = {
                            "train/loss": step_loss,
                            "train/lr": self.lr_scheduler.get_last_lr()[0],
                        }
                        self.log(logs, self.completed_steps)

                    if save_steps and self.completed_steps % save_steps == 0:
                        self.save_checkpoint(self.checkpoint_dir / f"step-{self.completed_steps}")

                    if eval_strategy == "steps" and eval_steps and self.completed_steps % eval_steps == 0:
                        eval_result = self.run_evaluation(self.completed_steps)
                        if self.is_main_process and eval_result:
                            last_eval = eval_result

            if eval_strategy == "epoch":
                eval_result = self.run_evaluation(self.completed_steps)
                if self.is_main_process and eval_result:
                    last_eval = eval_result

            if self.completed_steps >= self.max_train_steps:
                break

        progress_bar.close()

        train_runtime = time.time() - start_time
        metrics: Dict[str, float] = {}
        if self.is_main_process:
            avg_loss = running_loss / max(self.completed_steps, 1)
            total_batch_size = self.train_cfg.per_device_batch_size * self.grad_accum
            metrics.update(
                {
                    "train_loss": avg_loss,
                    "train_global_steps": float(self.completed_steps),
                    "train_runtime": train_runtime,
                    "train_samples_per_second": (total_batch_size * self.completed_steps) / max(train_runtime, 1e-8),
                    "train_steps_per_second": self.completed_steps / max(train_runtime, 1e-8),
                }
            )
            if last_eval:
                metrics.update({f"eval_{k}": v for k, v in last_eval.items()})
        
        return metrics

    def run_evaluation(self, step: int) -> Dict[str, float]:
        if hasattr(self.state.metric, "reset"):
            self.state.metric.reset()
        self.state.model.eval()
        eval_loss_total = 0.0
        eval_loss_steps = 0
        
        for batch in self.eval_dataloader:
            with torch.no_grad():
                outputs = self.state.model(**batch)
            
            loss = getattr(outputs, "loss", None)
            if loss is not None:
                loss_value = self.get_step_loss(loss).item()
                eval_loss_total += loss_value
                eval_loss_steps += 1
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            references = batch["labels"]
            
            predictions = self.gather_for_metrics(predictions)
            references = self.gather_for_metrics(references)
            
            if self.is_main_process:
                self.state.metric.add_batch(
                    predictions=predictions.cpu().numpy(),
                    references=references.cpu().numpy(),
                )
        
        self.wait_for_everyone()
        results: Dict[str, float] = {}
        if self.is_main_process:
            computed = self.state.metric.compute()
            results = {str(k): float(v) for k, v in computed.items()}
            if eval_loss_steps > 0:
                results.setdefault("loss", eval_loss_total / eval_loss_steps)
            self.log({f"eval_{k}": v for k, v in results.items()}, step)

            if self.metric_for_best_model and self.metric_for_best_model in results:
                current_metric_val = results[self.metric_for_best_model]
                if (self.metric_greater_is_better and current_metric_val > self.best_metric_val) or \
                   (not self.metric_greater_is_better and current_metric_val < self.best_metric_val):
                    self.best_metric_val = current_metric_val
                    self.save_checkpoint(self.best_model_dir)
                    self.log({f"best_{self.metric_for_best_model}": self.best_metric_val}, step)

        self.wait_for_everyone()
        self.state.model.train()
        return results if self.is_main_process else {}

    def save_final_model(self):
        if self.is_main_process:
            print(f"Saving final model to {self.output_dir}")
            unwrapped_model = self.unwrap_model()
            unwrapped_model.save_pretrained(self.output_dir, safe_serialization=True)
            self.state.tokenizer.save_pretrained(self.output_dir)
