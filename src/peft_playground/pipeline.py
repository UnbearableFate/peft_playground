"""Training pipeline utilities for PEFT experiments."""
from __future__ import annotations

import dataclasses
from contextlib import nullcontext
import datetime
import math
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from datasets import load_dataset
from evaluate import load as load_metric
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    get_scheduler,
    set_seed,
)

from .adapters import (
    AdapterConfig,
    SubspaceStrategy,
    attach_adapters,
    flora_factory,
    lora_factory,
    sva_factory,
    mam_factory,
)
from .config import TrainingConfig


METHOD_FACTORY = {
    "lora": lora_factory,
    "flora": flora_factory,
    "sva": sva_factory,
    "singular_value_adjustment": sva_factory,
    "mam": mam_factory,
}


def _convert_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    normalized = dtype.lower()
    mapping: Dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype}")
    return mapping[normalized]




def _maybe_init_wandb(cfg: TrainingConfig, model) -> Any:
    wandb_cfg = getattr(cfg, "wandb", None)
    if wandb_cfg is None or not wandb_cfg.enabled:
        return None
    try:
        import wandb  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "Weights & Biases logging requested but `wandb` is not installed. "
            "Install it with `pip install wandb`."
        ) from exc

    init_kwargs = {
        key: getattr(wandb_cfg, key)
        for key in ("project", "entity", "name", "group", "notes")
        if getattr(wandb_cfg, key) is not None
    }
    if wandb_cfg.tags:
        init_kwargs["tags"] = list(wandb_cfg.tags)
    if wandb_cfg.log_model:
        init_kwargs["log_model"] = wandb_cfg.log_model

    run = wandb.init(**init_kwargs)
    run.config.update(
        {
            "model_name_or_path": cfg.model.name_or_path,
            "dataset": cfg.dataset.name,
            "dataset_subset": cfg.dataset.subset,
            "adapter_method": cfg.adapter.method,
            "adapter_rank": cfg.adapter.rank,
        },
        allow_val_change=True,
    )
    if wandb_cfg.watch:
        log_freq = cfg.trainer.logging_steps or 100
        wandb.watch(model, log=wandb_cfg.watch, log_freq=log_freq)
    return run




def build_training_state(cfg: TrainingConfig, *, init_wandb: bool = True) -> Dict[str, Any]:
    """Prepare model, datasets, and shared artefacts without instantiating a Trainer."""
    print(f"Preparing training state for model {cfg.model.name_or_path}")
    set_seed(cfg.runtime.seed)

    tokenizer = _prepare_tokenizer(cfg)

    raw_dataset = load_dataset(cfg.dataset.name, cfg.dataset.subset)
    train_split = raw_dataset[cfg.dataset.train_split]
    label_feature = train_split.features[cfg.dataset.label_field]

    if hasattr(label_feature, "names") and label_feature.names is not None:
        label_list = list(label_feature.names)
    else:
        unique_labels = sorted(set(train_split[cfg.dataset.label_field]))
        label_list = [str(label) for label in unique_labels]

    finetuning_task = cfg.dataset.subset or cfg.dataset.name
    num_labels = len(label_list)

    dtype = _convert_dtype(cfg.model.torch_dtype)
    auto_config = AutoConfig.from_pretrained(
        cfg.model.name_or_path,
        revision=cfg.model.revision,
        num_labels=num_labels,
        finetuning_task=finetuning_task,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    model_kwargs: Dict[str, Any] = {
        "config": auto_config,
        "revision": cfg.model.revision,
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if cfg.model.device_map is not None:
        model_kwargs["device_map"] = cfg.model.device_map

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        **model_kwargs,
    )

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.runtime.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    wandb_run = _maybe_init_wandb(cfg, model) if init_wandb else None

    adapter_config = _build_adapter_config(cfg)
    method = cfg.adapter.method.lower()
    if method not in METHOD_FACTORY:
        raise ValueError(f"Unsupported adapter method: {cfg.adapter.method}")
    factory = METHOD_FACTORY[method]
    attach_adapters(model, factory, adapter_config)

    train_dataset, eval_dataset = _prepare_datasets(cfg, raw_dataset, tokenizer, label_list)
    data_collator = DataCollatorWithPadding(tokenizer)

    metric = load_metric("glue", cfg.dataset.subset) if cfg.dataset.subset else load_metric(cfg.dataset.name)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "metric": metric,
        "label_list": label_list,
        "wandb_run": wandb_run,
    }


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _gather_tensor_for_metrics(tensor: torch.Tensor) -> torch.Tensor:
    cpu_tensor = tensor.detach().cpu()
    if not _is_distributed():
        return cpu_tensor
    world_size = dist.get_world_size()
    tensor_list = [None] * world_size
    dist.all_gather_object(tensor_list, cpu_tensor)
    return torch.cat(tensor_list, dim=0)


def _distributed_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not _is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


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


def _build_adapter_config(cfg: TrainingConfig) -> AdapterConfig:
    method = cfg.adapter.method.lower()
    if method in {"sva", "singular_value_adjustment"}:
        strategy = SubspaceStrategy.RECONSTRUCTION
    else:
        strategy = SubspaceStrategy.EXTENSION
    return AdapterConfig(
        target_modules=cfg.adapter.target_modules,
        rank=cfg.adapter.rank,
        alpha=cfg.adapter.alpha,
        dropout=cfg.adapter.dropout,
        init_scale=cfg.adapter.init_scale,
        strategy=strategy,
        train_bias=cfg.adapter.train_bias,
        name=cfg.adapter.method,
        extra=dict(cfg.adapter.extra),
    )


def _prepare_tokenizer(cfg: TrainingConfig):
    model_cfg = cfg.model
    tokenizer_name = model_cfg.tokenizer_name or model_cfg.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=model_cfg.revision,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def _prepare_datasets(cfg: TrainingConfig, raw_dataset, tokenizer, label_list: List[str]):
    ds_cfg = cfg.dataset
    dataset = raw_dataset

    label_to_id = None
    if label_list:
        label_to_id = {}
        for idx, label in enumerate(label_list):
            keys = {label, str(label), idx, str(idx)}
            for key in keys:
                label_to_id[key] = idx

    text_fields = list(ds_cfg.text_fields)
    label_field = ds_cfg.label_field
    max_length = ds_cfg.max_length

    def preprocess_examples(examples: Mapping[str, Any]) -> Mapping[str, Any]:
        if len(text_fields) == 1:
            model_inputs = tokenizer(
                examples[text_fields[0]],
                truncation=True,
                max_length=max_length,
            )
        elif len(text_fields) == 2:
            model_inputs = tokenizer(
                examples[text_fields[0]],
                examples[text_fields[1]],
                truncation=True,
                max_length=max_length,
            )
        else:
            raise ValueError("Tokenizer currently supports up to two text fields")

        labels = examples[label_field]
        if label_to_id is not None:
            def map_label(value):
                if value in label_to_id:
                    return label_to_id[value]
                value_str = str(value)
                return label_to_id.get(value_str, value)

            if isinstance(labels, list):
                model_inputs["labels"] = [map_label(label) for label in labels]
            else:
                model_inputs["labels"] = map_label(labels)
        else:
            model_inputs["labels"] = labels
        return model_inputs

    train_dataset = dataset[ds_cfg.train_split]
    eval_dataset = dataset[ds_cfg.eval_split]

    remove_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        preprocess_examples,
        batched=True,
        remove_columns=remove_columns,
    )
    eval_dataset = eval_dataset.map(
        preprocess_examples,
        batched=True,
        remove_columns=remove_columns,
    )
    return train_dataset, eval_dataset


def prepare_training_components(cfg: TrainingConfig):
    state = build_training_state(cfg)

    model = state["model"]
    tokenizer = state["tokenizer"]
    train_dataset = state["train_dataset"]
    eval_dataset = state["eval_dataset"]
    data_collator = state["data_collator"]
    metric = state["metric"]
    wandb_run = state["wandb_run"]

    def compute_metrics(eval_pred):
        if hasattr(eval_pred, "predictions"):
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    trainer_args_dict = dataclasses.asdict(cfg.trainer)
    extra = trainer_args_dict.pop("extra", {})
    trainer_args_dict.update(extra)
    report_to = list(trainer_args_dict.get("report_to", []))
    if cfg.wandb.enabled:
        report_to = [dest for dest in report_to if dest != "none"]
        if "wandb" not in report_to:
            report_to.append("wandb")
        if cfg.wandb.name and not trainer_args_dict.get("run_name"):
            trainer_args_dict["run_name"] = cfg.wandb.name
    trainer_args_dict["report_to"] = report_to
    if "evaluation_strategy" in trainer_args_dict and "eval_strategy" not in trainer_args_dict:
        trainer_args_dict["eval_strategy"] = trainer_args_dict.pop("evaluation_strategy")
    if trainer_args_dict.get("max_steps") is None:
        trainer_args_dict.pop("max_steps", None)
    if trainer_args_dict.get("save_steps") is None:
        trainer_args_dict.pop("save_steps", None)
    if trainer_args_dict.get("eval_steps") is None:
        trainer_args_dict.pop("eval_steps", None)
    training_args = TrainingArguments(**trainer_args_dict)

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

    components = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "trainer": trainer,
        "wandb_run": wandb_run,
    }

    return components


def run_accelerated_training(
    cfg: TrainingConfig,
    *,
    evaluate_only: bool = False,
    dry_run: bool = False,
):
    """Train/evaluate using Hugging Face Accelerate for DDP."""

    grad_accum = cfg.accelerate.gradient_accumulation_steps or cfg.trainer.gradient_accumulation_steps
    grad_accum = max(int(grad_accum or 1), 1)

    log_with = cfg.accelerate.log_with if cfg.accelerate.log_with else None

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=cfg.accelerate.mixed_precision,
        log_with=log_with,
        project_dir=cfg.accelerate.project_dir,
        split_batches=cfg.accelerate.split_batches,
        even_batches=cfg.accelerate.even_batches,
    )

    state = build_training_state(cfg, init_wandb=accelerator.is_main_process and cfg.wandb.enabled)

    model = state["model"]
    tokenizer = state["tokenizer"]
    train_dataset = state["train_dataset"]
    eval_dataset = state["eval_dataset"]
    data_collator = state["data_collator"]
    metric = state["metric"]
    wandb_run = state["wandb_run"] if accelerator.is_main_process else None

    train_batch_size = cfg.trainer.per_device_train_batch_size
    eval_batch_size = cfg.trainer.per_device_eval_batch_size
    num_workers = int(cfg.trainer.extra.get("dataloader_num_workers", 0))
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=eval_batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
    )

    train_dataloader_len = max(1, len(train_dataloader))
    num_update_steps_per_epoch = math.ceil(train_dataloader_len / grad_accum)
    if cfg.trainer.max_steps:
        max_train_steps = int(cfg.trainer.max_steps)
    else:
        max_train_steps = int(math.ceil(cfg.trainer.num_train_epochs * num_update_steps_per_epoch))
    num_epochs = max(1, math.ceil(max_train_steps / num_update_steps_per_epoch))

    warmup_steps = int(cfg.trainer.warmup_ratio * max_train_steps)
    scheduler_name = cfg.trainer.extra.get("lr_scheduler_type", "linear")
    lr_scheduler = get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    (model, optimizer, train_dataloader, eval_dataloader, lr_scheduler) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    )

    output_dir = Path(cfg.trainer.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if dry_run:
        accelerator.print(
            "Dry run: Accelerate prepared with "
            f"world_size={accelerator.state.num_processes}, gradient_accumulation={grad_accum}."
        )
        return {}, wandb_run

    def log_metrics(logs: Dict[str, float], step: int) -> None:
        if not logs:
            return
        if accelerator.is_main_process:
            accelerator.log(logs, step=step)
            if wandb_run is not None:
                wandb_run.log(logs, step=step)

    def run_evaluation(step: int) -> Dict[str, float]:  # returns metrics on main process
        if hasattr(metric, "reset"):
            metric.reset()
        model.eval()
        eval_loss_total = 0.0
        eval_loss_steps = 0
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            loss = getattr(outputs, "loss", None)
            if loss is not None:
                loss_value = accelerator.reduce(loss.detach(), reduction="mean").item()
                eval_loss_total += loss_value
                eval_loss_steps += 1
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            references = batch["labels"]
            predictions = accelerator.gather_for_metrics(predictions)
            references = accelerator.gather_for_metrics(references)
            if accelerator.is_main_process:
                metric.add_batch(
                    predictions=predictions.cpu().numpy(),
                    references=references.cpu().numpy(),
                )
        accelerator.wait_for_everyone()
        results: Dict[str, float] = {}
        if accelerator.is_main_process:
            computed = metric.compute()
            results = {str(k): float(v) for k, v in computed.items()}
            if eval_loss_steps > 0:
                results.setdefault("loss", eval_loss_total / eval_loss_steps)
            log_metrics({f"eval_{k}": v for k, v in results.items()}, step)
        accelerator.wait_for_everyone()
        model.train()
        return results if accelerator.is_main_process else {}

    if evaluate_only:
        eval_results = run_evaluation(step=0)
        return ({f"eval_{k}": v for k, v in eval_results.items()}, wandb_run)

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("training")

    completed_steps = 0
    running_loss = 0.0
    logging_steps = int(cfg.trainer.logging_steps or 0)
    logged_steps = 0
    eval_strategy = (cfg.trainer.eval_strategy or "no").lower()
    eval_steps = cfg.trainer.eval_steps
    start_time = time.time()
    last_eval: Dict[str, float] = {}

    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.update(1)
                step_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
                running_loss += step_loss
                logged_steps += 1

                if logging_steps and completed_steps % logging_steps == 0:
                    logs = {
                        "train/loss": step_loss,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                    }
                    log_metrics(logs, completed_steps)

                if eval_strategy == "steps" and eval_steps and completed_steps % eval_steps == 0:
                    eval_result = run_evaluation(completed_steps)
                    if accelerator.is_main_process and eval_result:
                        last_eval = eval_result

                if completed_steps >= max_train_steps:
                    break

        if eval_strategy == "epoch" and not evaluate_only:
            eval_result = run_evaluation(completed_steps)
            if accelerator.is_main_process and eval_result:
                last_eval = eval_result

        if completed_steps >= max_train_steps:
            break

    progress_bar.close()

    train_runtime = time.time() - start_time
    metrics: Dict[str, float] = {}
    if accelerator.is_main_process:
        avg_loss = running_loss / max(logged_steps, 1)
        total_batch_size = (
            cfg.trainer.per_device_train_batch_size
            * accelerator.state.num_processes
            * grad_accum
        )
        metrics.update(
            {
                "train_loss": avg_loss,
                "train_global_steps": float(completed_steps),
                "train_runtime": train_runtime,
                "train_samples_per_second": (total_batch_size * completed_steps) / max(train_runtime, 1e-8),
                "train_steps_per_second": completed_steps / max(train_runtime, 1e-8),
            }
        )
        if last_eval:
            metrics.update({f"eval_{k}": v for k, v in last_eval.items()})

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if not evaluate_only:
        if accelerator.is_main_process:
            unwrapped.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
    accelerator.wait_for_everyone()

    return (metrics if accelerator.is_main_process else {}, wandb_run)


def run_ddp_training(
    cfg: TrainingConfig,
    *,
    evaluate_only: bool = False,
    dry_run: bool = False,
):
    """Train/evaluate using native torch.distributed DDP."""
    print("DDP training requested")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if not _is_distributed():
        backend = cfg.ddp.backend or "nccl"
        init_method = cfg.ddp.init_method or "env://"
        mpi_rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
        mpi_world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1)))
        assert mpi_world_size > 1, "DDP requested but only one process launched"
        print(f"Initializing DDP: rank {mpi_rank}/{mpi_world_size}, backend={backend}, init_method={init_method}")
        dist.init_process_group(backend=backend, init_method=init_method, 
                                rank=mpi_rank, world_size=mpi_world_size,
                                timeout=datetime.timedelta(minutes=20), device_id=device)

    rank = dist.get_rank() if _is_distributed() else 0
    world_size = dist.get_world_size() if _is_distributed() else 1
    #local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("MPI_LOCALRANKID", 0)))
    
    grad_accum = cfg.ddp.gradient_accumulation_steps or cfg.trainer.gradient_accumulation_steps
    grad_accum = max(int(grad_accum or 1), 1)

    state = build_training_state(cfg, init_wandb=(rank == 0 and cfg.wandb.enabled))

    model = state["model"].to(device)
    tokenizer = state["tokenizer"]
    train_dataset = state["train_dataset"]
    eval_dataset = state["eval_dataset"]
    data_collator = state["data_collator"]
    metric = state["metric"]
    wandb_run = state["wandb_run"] if rank == 0 else None

    ddp_kwargs = {
        "find_unused_parameters": cfg.ddp.find_unused_parameters,
        "broadcast_buffers": cfg.ddp.broadcast_buffers,
        "static_graph": cfg.ddp.static_graph,
    }
    if device.type == "cuda":
        ddp_kwargs["device_ids"] = [device.index]
        ddp_kwargs["output_device"] = device.index

    model = DDP(model, **ddp_kwargs)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    num_workers = int(cfg.trainer.extra.get("dataloader_num_workers", 0))
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.trainer.per_device_train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.trainer.per_device_eval_batch_size,
        sampler=eval_sampler,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
    )

    train_loader_len = max(1, len(train_loader))
    num_update_steps_per_epoch = math.ceil(train_loader_len / grad_accum)
    if cfg.trainer.max_steps:
        max_train_steps = int(cfg.trainer.max_steps)
        num_epochs = max(1, math.ceil(max_train_steps / num_update_steps_per_epoch))
    else:
        num_epochs = max(int(math.ceil(cfg.trainer.num_train_epochs)), 1)
        max_train_steps = num_epochs * num_update_steps_per_epoch

    warmup_steps = int(cfg.trainer.warmup_ratio * max_train_steps)
    scheduler_name = cfg.trainer.extra.get("lr_scheduler_type", "linear")
    lr_scheduler = get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    if dry_run:
        if rank == 0:
            print(
                "Dry run: DDP prepared with world_size="
                f"{world_size}, gradient_accumulation={grad_accum}"
            )
        return {}, wandb_run

    def log_metrics(logs: Dict[str, float], step: int) -> None:
        if logs and rank == 0:
            if wandb_run is not None:
                wandb_run.log(logs, step=step)
            print(f"[ddp] step {step}: {logs}")

    def run_evaluation(step: int) -> Dict[str, float]:
        if hasattr(metric, "reset"):
            metric.reset()
        model.eval()
        eval_loss_total = 0.0
        eval_loss_steps = 0
        eval_sampler.set_epoch(step if step else 0)
        with torch.no_grad():
            for batch in eval_loader:
                batch = _move_batch_to_device(batch, device)
                outputs = model(**batch)
                loss = getattr(outputs, "loss", None)
                if loss is not None:
                    reduced = _distributed_mean(loss.detach())
                    eval_loss_total += reduced.item()
                    eval_loss_steps += 1
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                gathered_preds = _gather_tensor_for_metrics(predictions)
                gathered_labels = _gather_tensor_for_metrics(batch["labels"])
                if rank == 0:
                    metric.add_batch(
                        predictions=gathered_preds.numpy(),
                        references=gathered_labels.numpy(),
                    )
        results: Dict[str, float] = {}
        if rank == 0:
            computed = metric.compute()
            results = {str(k): float(v) for k, v in computed.items()}
            if eval_loss_steps > 0:
                results.setdefault("loss", eval_loss_total / max(eval_loss_steps, 1))
            log_metrics({f"eval_{k}": v for k, v in results.items()}, step)
        dist.barrier()
        model.train()
        return results

    total_batch_size = (
        cfg.trainer.per_device_train_batch_size
        * world_size
        * grad_accum
    )

    running_loss = 0.0
    logged_steps = 0
    completed_steps = 0
    logging_steps = int(cfg.trainer.logging_steps or 0)
    eval_strategy = (cfg.trainer.eval_strategy or "no").lower()
    eval_steps = cfg.trainer.eval_steps
    last_eval: Dict[str, float] = {}
    start_time = time.time()
    print(f"Starting training for {num_epochs} epochs at {time.localtime()}, dataset length: {len(train_loader)}")
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, device)
            sync_context = (
                nullcontext()
                if (grad_accum == 1 or (step + 1) % grad_accum == 0)
                else model.no_sync()
            )
            with sync_context:
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
                loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

                reduced_loss = _distributed_mean(outputs.loss.detach())
                running_loss += reduced_loss.item()
                logged_steps += 1

                if logging_steps and completed_steps % logging_steps == 0:
                    log_metrics(
                        {
                            "train/loss": reduced_loss.item(),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                        },
                        completed_steps,
                    )

                if eval_strategy == "steps" and eval_steps and completed_steps % eval_steps == 0:
                    eval_result = run_evaluation(completed_steps)
                    if rank == 0 and eval_result:
                        last_eval = eval_result

                if completed_steps >= max_train_steps:
                    break

        if completed_steps >= max_train_steps:
            break

        if eval_strategy == "epoch" and not evaluate_only:
            eval_result = run_evaluation(completed_steps)
            if rank == 0 and eval_result:
                last_eval = eval_result

    dist.barrier()
    train_runtime = time.time() - start_time

    metrics: Dict[str, float] = {}
    if rank == 0:
        avg_loss = running_loss / max(logged_steps, 1)
        metrics.update(
            {
                "train_loss": avg_loss,
                "train_global_steps": float(completed_steps),
                "train_runtime": train_runtime,
                "train_samples_per_second": (total_batch_size * completed_steps) / max(train_runtime, 1e-8),
                "train_steps_per_second": completed_steps / max(train_runtime, 1e-8),
            }
        )
        if last_eval:
            metrics.update({f"eval_{k}": v for k, v in last_eval.items()})

    dist.barrier()

    if not evaluate_only and rank == 0:
        output_dir = Path(cfg.trainer.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module
        unwrapped.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

    dist.barrier()

    return (metrics if rank == 0 else {}, wandb_run)
