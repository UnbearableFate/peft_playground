"""Monitoring callbacks for training."""
from __future__ import annotations

from transformers import TrainerCallback
import torch


class TrainingMonitorCallback(TrainerCallback):
    """Logs gradient norms and GPU usage to the Trainer logging stream."""

    def __init__(self, log_grad_norm: bool = True, log_gpu_stats: bool = True) -> None:
        self.log_grad_norm = log_grad_norm
        self.log_gpu_stats = log_gpu_stats

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: D401
        if self.log_gpu_stats and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: D401
        trainer = kwargs.get("trainer")
        model = kwargs.get("model")
        logs = {}
        if self.log_grad_norm and model is not None:
            grad_norm = self._compute_grad_norm(model)
            if grad_norm is not None:
                logs["train/grad_norm"] = grad_norm
        if self.log_gpu_stats and torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logs["system/gpu_mem_max_mb"] = max_mem
            torch.cuda.reset_peak_memory_stats()
        if logs and trainer is not None:
            trainer.log(logs)
        return control

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: D401
        trainer = kwargs.get("trainer")
        logs = {}
        if self.log_gpu_stats and torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logs["eval/gpu_mem_max_mb"] = max_mem
            torch.cuda.reset_peak_memory_stats()
        if logs and trainer is not None:
            trainer.log(logs)
        return control

    @staticmethod
    def _compute_grad_norm(model) -> float | None:
        total_norm_sq = 0.0
        has_grad = False
        for param in model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce()
            norm = grad.norm(2)
            total_norm_sq += norm.item() ** 2
            has_grad = True
        if not has_grad:
            return None
        return total_norm_sq ** 0.5
