"""Training pipeline utilities for PEFT experiments."""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)

from .adapters import AdapterConfig, SubspaceStrategy, attach_adapters, flora_factory, lora_factory
from .config import TrainingConfig


METHOD_FACTORY = {
    "lora": lora_factory,
    "flora": flora_factory,
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
    set_seed(cfg.runtime.seed)

    tokenizer = _prepare_tokenizer(cfg)

    # Load dataset first to infer label space before instantiating the model.
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
        model_kwargs["torch_dtype"] = dtype
    if cfg.model.device_map is not None:
        model_kwargs["device_map"] = cfg.model.device_map

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        **model_kwargs,
    )

    if getattr(model.config, 'pad_token_id', None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.runtime.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    wandb_run = _maybe_init_wandb(cfg, model)

    adapter_config = _build_adapter_config(cfg)
    method = cfg.adapter.method.lower()
    if method not in METHOD_FACTORY:
        raise ValueError(f"Unsupported adapter method: {cfg.adapter.method}")

    factory = METHOD_FACTORY[method]
    attach_adapters(model, factory, adapter_config)

    train_dataset, eval_dataset = _prepare_datasets(cfg, raw_dataset, tokenizer, label_list)
    data_collator = DataCollatorWithPadding(tokenizer)

    metric = load_metric("glue", cfg.dataset.subset) if cfg.dataset.subset else load_metric(cfg.dataset.name)

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

