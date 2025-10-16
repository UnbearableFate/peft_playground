"""Runner that relies on Hugging Face Trainer."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from peft_playground.training.monitors import TrainingMonitorCallback

from ...config import TrainingConfig
from ..preparation import build_training_state


def run_trainer(
    cfg: TrainingConfig,
    *,
    evaluate_only: bool = False,
    dry_run: bool = False,
) -> tuple[Dict[str, Any], Any]:
    """Execute training/evaluation with the standard Trainer backend."""

    components = prepare_training_components(cfg)
    trainer = components["trainer"]
    wandb_run = components.get("wandb_run")

    if dry_run:
        return {"trainer_args": trainer.args.to_dict()}, wandb_run

    metrics: Dict[str, Any] = {}
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

    return metrics, wandb_run
    
def prepare_training_components(cfg: TrainingConfig):
    state = build_training_state(cfg)
    model = state.model
    tokenizer = state.tokenizer
    train_dataset = state.train_dataset
    eval_dataset = state.eval_dataset
    data_collator = state.data_collator
    metric = state.metric
    wandb_run = state.wandb_run

    def compute_metrics(eval_pred):
        if hasattr(eval_pred, "predictions"):
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    train_cfg = cfg.train
    eval_cfg = cfg.evaluation

    training_args_dict: Dict[str, Any] = {
        "output_dir": str(train_cfg.output_dir),
        "num_train_epochs": train_cfg.num_epochs,
        "per_device_train_batch_size": train_cfg.per_device_batch_size,
        "per_device_eval_batch_size": eval_cfg.per_device_batch_size,
        "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
        "warmup_ratio": train_cfg.warmup_ratio,
        "learning_rate": train_cfg.learning_rate,
        "weight_decay": train_cfg.weight_decay,
        "logging_steps": train_cfg.logging_steps,
        "max_steps": train_cfg.max_steps,
        "lr_scheduler_type": train_cfg.lr_scheduler_type,
        "eval_strategy": eval_cfg.strategy,
        "eval_steps": eval_cfg.steps,
        "save_strategy": eval_cfg.save_strategy,
        "save_steps": eval_cfg.save_steps,
        "load_best_model_at_end": eval_cfg.load_best_model_at_end,
        "metric_for_best_model": eval_cfg.metric_for_best_model,
        "greater_is_better": eval_cfg.greater_is_better,
        "do_train": eval_cfg.do_train,
        "do_eval": eval_cfg.do_eval,
        "save_total_limit": eval_cfg.save_total_limit,
        "fp16": train_cfg.precision.fp16,
        "bf16": train_cfg.precision.bf16,
        "push_to_hub": train_cfg.push_to_hub,
    }

    if train_cfg.run_name is not None:
        training_args_dict["run_name"] = train_cfg.run_name
    if train_cfg.resume_from_checkpoint is not None:
        training_args_dict["resume_from_checkpoint"] = train_cfg.resume_from_checkpoint
    if train_cfg.extra:
        training_args_dict.update(train_cfg.extra)
    if eval_cfg.extra:
        training_args_dict.update(eval_cfg.extra)

    report_to = list(train_cfg.report_to)
    if cfg.wandb.enabled:
        report_to = [dest for dest in report_to if dest != "none"]
        if "wandb" not in report_to:
            report_to.append("wandb")
        if cfg.wandb.name and "run_name" not in training_args_dict:
            training_args_dict["run_name"] = cfg.wandb.name
    training_args_dict["report_to"] = report_to or ["none"]

    if training_args_dict.get("max_steps") is None:
        training_args_dict.pop("max_steps", None)
    if training_args_dict.get("save_steps") is None:
        training_args_dict.pop("save_steps", None)
    if training_args_dict.get("eval_steps") is None:
        training_args_dict.pop("eval_steps", None)
    if training_args_dict.get("save_total_limit") is None:
        training_args_dict.pop("save_total_limit", None)

    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(
        TrainingMonitorCallback(
            log_grad_norm=True,
            log_gpu_stats=torch.cuda.is_available(),
        )
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "trainer": trainer,
        "wandb_run": wandb_run,
    }
