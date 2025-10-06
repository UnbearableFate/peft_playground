"""Model/dataset preparation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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
    set_seed,
)

from ..adapters import (
    AdapterConfig,
    SubspaceStrategy,
    attach_adapters,
    dora_factory,
    flora_factory,
    lora_factory,
    mam_factory,
    sva_factory,
)
from ..config import TrainingConfig
from .monitors import TrainingMonitorCallback
from .state import TrainingState

METHOD_FACTORY = {
    "lora": lora_factory,
    "flora": flora_factory,
    "dora": dora_factory,
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
    except ImportError as exc:  # pragma: no cover
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
        log_freq = cfg.train.logging_steps
        log_freq = log_freq or 100
        wandb.watch(model, log=wandb_cfg.watch, log_freq=log_freq)
    return run


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


def _build_adapter_config(cfg: TrainingConfig) -> AdapterConfig:
    method = cfg.adapter.method.lower()
    if method in {"sva", "singular_value_adjustment"}:
        strategy = SubspaceStrategy.RECONSTRUCTION
    elif method == "dora":
        strategy = SubspaceStrategy.COMBINATION
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


def build_training_state(cfg: TrainingConfig, *, init_wandb: bool = True) -> TrainingState:
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
        model_kwargs["torch_dtype"] = dtype
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

    return TrainingState(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        metric=metric,
        label_list=label_list,
        wandb_run=wandb_run,
    )