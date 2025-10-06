"""Native torch.distributed DDP runner."""
from __future__ import annotations

import datetime
import os
from contextlib import nullcontext
from pathlib import Path
import time
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ...adapters import load_attached_adapters, serialize_attached_adapters
from ...config import TrainingConfig
from ..utils import distributed_mean, gather_tensor_for_metrics, is_distributed, move_batch_to_device
from .base_runner import BaseRunner


class DDPRunner(BaseRunner):
    """Runner for training with native torch DDP."""

    def setup(self) -> None:
        if not is_distributed():
            backend = self.cfg.ddp.backend or "nccl"
            init_method = self.cfg.ddp.init_method or "env://"
            mpi_world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
            mpi_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
            assert mpi_world_size > 0 and mpi_rank >= 0, "MPI environment variables not set"
            dist.init_process_group(
                backend=backend, 
                init_method=init_method, 
                world_size=mpi_world_size, 
                rank=mpi_rank, 
                timeout=datetime.timedelta(minutes=30)
            )

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.grad_accum = self.cfg.ddp.gradient_accumulation_steps or self.train_cfg.gradient_accumulation_steps
        self.grad_accum = max(int(self.grad_accum or 1), 1)

    def create_dataloaders(self) -> None:
        """Create DDP-specific dataloaders with DistributedSampler."""
        train_sampler = DistributedSampler(
            self.state.train_dataset, 
            num_replicas=dist.get_world_size(), 
            rank=dist.get_rank(), 
            shuffle=True
        )
        eval_sampler = DistributedSampler(
            self.state.eval_dataset, 
            num_replicas=dist.get_world_size(), 
            rank=dist.get_rank(), 
            shuffle=False
        )

        num_workers = int(self.train_cfg.dataloader_num_workers or 0)
        pin_memory = self.device.type == "cuda"

        self.train_dataloader = DataLoader(
            self.state.train_dataset,
            batch_size=self.train_cfg.per_device_batch_size,
            sampler=train_sampler,
            collate_fn=self.state.data_collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.eval_dataloader = DataLoader(
            self.state.eval_dataset,
            batch_size=self.eval_cfg.per_device_batch_size,
            sampler=eval_sampler,
            collate_fn=self.state.data_collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def prepare(self) -> None:
        self.state.model = self.state.model.to(self.device)
        ddp_kwargs = {
            "find_unused_parameters": self.cfg.ddp.find_unused_parameters,
            "broadcast_buffers": self.cfg.ddp.broadcast_buffers,
            "static_graph": self.cfg.ddp.static_graph,
        }
        if self.device.type == "cuda":
            ddp_kwargs.update(device_ids=[self.device.index], output_device=self.device.index)

        self.state.model = DDP(self.state.model, **ddp_kwargs)

    def backward(self, loss: torch.Tensor) -> None:
        # The loss is already scaled by grad_accum in the training loop
        loss.backward()

    def get_step_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return distributed_mean(loss.detach())

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        return gather_tensor_for_metrics(tensor)

    @property
    def is_main_process(self) -> bool:
        return dist.get_rank() == 0

    def wait_for_everyone(self) -> None:
        dist.barrier()

    def unwrap_model(self) -> torch.nn.Module:
        return self.state.model.module

    def log(self, data: Dict[str, Any], step: int) -> None:
        if self.is_main_process:
            if self.state.wandb_run:
                self.state.wandb_run.log(data, step=step)
            print(f"[ddp] step {step}: {data}")

    def save_checkpoint(self, output_dir: Path) -> None:
        if self.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            unwrapped_model = self.unwrap_model()

            checkpoint = {
                'adapters': serialize_attached_adapters(unwrapped_model),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'completed_steps': self.completed_steps,
                'starting_epoch': self.current_epoch,
                'best_metric_val': self.best_metric_val,
            }

            torch.save(checkpoint, output_dir / 'training_state.pt')
            self.state.tokenizer.save_pretrained(output_dir)
            print(f"Checkpoint saved to {output_dir}")

    def load_checkpoint(self, resume_from: Path) -> None:
        checkpoint = torch.load(resume_from / 'training_state.pt', map_location=self.device)

        model = self.unwrap_model()
        adapter_state = checkpoint.get('adapters')
        if adapter_state is not None:
            summary = load_attached_adapters(model, adapter_state)
            if self.is_main_process:
                if summary["missing"]:
                    print(f"Warning: adapters missing during load and skipped: {summary['missing']}")
                if summary["incompatible"]:
                    print(f"Warning: adapter configs mismatched and skipped: {summary['incompatible']}")
        elif 'model' in checkpoint:
            # Backwards compatibility with checkpoints saved before adapter-only format.
            model.load_state_dict(checkpoint['model'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.completed_steps = checkpoint.get('completed_steps', 0)
        self.starting_epoch = checkpoint.get('starting_epoch', 0)
        self.best_metric_val = checkpoint.get('best_metric_val', self.best_metric_val)

        self.wait_for_everyone()
        print(f"Resumed from checkpoint: {resume_from}. Starting at epoch {self.starting_epoch}, step {self.completed_steps}")

    def run_training(self) -> Dict[str, float]:
        # Override to handle batch moving and no_sync context
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
            self.train_dataloader.sampler.set_epoch(epoch)
            self.current_epoch = epoch
            
            for step, batch in enumerate(self.train_dataloader):
                batch = move_batch_to_device(batch, self.device)
                
                sync_context = (
                    self.state.model.no_sync()
                    if self.grad_accum > 1 and (step + 1) % self.grad_accum != 0
                    else nullcontext()
                )

                with sync_context:
                    outputs = self.state.model(**batch)
                    loss = outputs.loss / self.grad_accum
                    self.backward(loss)
                
                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(self.train_dataloader):
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.completed_steps += 1
                    progress_bar.update(1)

                    step_loss = self.get_step_loss(outputs.loss).item()
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

                if self.completed_steps >= self.max_train_steps:
                    break
            
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
            total_batch_size = (
                self.train_cfg.per_device_batch_size
                * dist.get_world_size()
                * self.grad_accum
            )
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
        self.eval_dataloader.sampler.set_epoch(step if step else 0)
        # Manually move batch to device
        original_eval_dataloader = self.eval_dataloader
        self.eval_dataloader = (move_batch_to_device(b, self.device) for b in original_eval_dataloader)
        eval_metrics = super().run_evaluation(step)
        self.eval_dataloader = original_eval_dataloader
        return eval_metrics


def run_ddp_training(
    cfg: TrainingConfig,
    *,
    evaluate_only: bool = False,
    dry_run: bool = False,
) -> tuple[Dict[str, Any], Any]:
    runner = DDPRunner(cfg, evaluate_only=evaluate_only, dry_run=dry_run)
    return runner.run()
