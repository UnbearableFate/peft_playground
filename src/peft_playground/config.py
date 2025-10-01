"""YAML-driven configuration for PEFT experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml


@dataclass
class ModelConfig:
    """Checkpoint loading options for the backbone model/tokenizer."""

    name_or_path: str  # Hugging Face repo id or local directory with weights
    revision: Optional[str] = None  # Optional git revision pinned when using the Hub
    torch_dtype: Optional[str] = None  # Text alias converted to torch.dtype (e.g. bfloat16)
    trust_remote_code: bool = False  # Allow custom modeling code from the repository
    device_map: Optional[Any] = None  # Passed through to from_pretrained for sharding/offloading
    tokenizer_name: Optional[str] = None  # Override tokenizer source if it differs from the model


@dataclass
class DatasetConfig:
    """Dataset selection and preprocessing knobs."""

    name: str  # datasets.load_dataset identifier
    subset: Optional[str] = None  # Configuration subset (e.g. mrpc, cola)
    train_split: str = "train"  # Split tag for training data
    eval_split: str = "validation"  # Split tag for evaluation data
    text_fields: Sequence[str] = field(default_factory=lambda: ["sentence"])  # Columns concatenated as input
    label_field: str = "label"  # Column containing supervision targets
    max_length: int = 512  # Tokenizer truncation length


@dataclass
class AdapterSettings:
    """LoRA/FLoRA-style adapter hyper-parameters."""

    method: str  # Adapter family key (lora, flora, ...)
    target_modules: Sequence[str]  # Module name suffixes for nn.Linear layers to wrap
    rank: int  # Low-rank dimensionality r
    alpha: float  # Scaling factor applied to the adapter residual
    dropout: float = 0.0  # Optional dropout on adapter inputs
    init_scale: float = 1.0  # Post-init multiplicative factor for adapter weights
    train_bias: bool = False  # Enable training of the wrapped linear biases
    extra: Dict[str, Any] = field(default_factory=dict)  # Free-form payload consumed by specific adapters


@dataclass
class TrainerConfig:
    """Arguments forwarded to Hugging Face TrainingArguments."""

    output_dir: str  # Where checkpoints, logs, and summaries are written
    num_train_epochs: float = 3.0  # Epoch budget when max_steps is unset
    per_device_train_batch_size: int = 8  # Micro-batch size for training
    per_device_eval_batch_size: int = 8  # Micro-batch size for evaluation
    gradient_accumulation_steps: int = 1  # Steps to accumulate gradients before optimizer step
    warmup_ratio: float = 0.0  # Fraction of steps used for LR warmup
    learning_rate: float = 5e-5  # Peak learning rate
    weight_decay: float = 0.0  # L2 regularisation strength
    eval_strategy: str = "steps"  # Evaluation cadence keyword
    eval_steps: Optional[int] = None  # Step interval for eval when using step strategy
    save_strategy: str = "steps"  # Checkpoint cadence keyword
    save_steps: Optional[int] = None  # Step interval for checkpointing under step strategy
    logging_steps: int = 50  # Logging frequency for Trainer callbacks/integrations
    max_steps: Optional[int] = None  # Hard limit on optimizer steps (overrides epochs)
    fp16: bool = False  # Enable fp16 mixed precision
    bf16: bool = False  # Enable bf16 mixed precision
    report_to: Sequence[str] = field(default_factory=lambda: ["none"])  # Reporting integrations (e.g. wandb)
    extra: Dict[str, Any] = field(default_factory=dict)  # Additional kwargs forwarded to TrainingArguments


@dataclass
class RuntimeConfig:
    """Runtime toggles applied outside the Trainer arguments."""

    seed: int = 42  # Global random seed fed into transformers.set_seed
    gradient_checkpointing: bool = False  # Enable model gradient checkpointing
    torch_compile: bool = False  # Placeholder for optional torch.compile integration


@dataclass
class WandbConfig:
    """Weights & Biases run metadata."""

    enabled: bool = False  # Toggle W&B logging on/off
    project: Optional[str] = None  # W&B project slug
    entity: Optional[str] = None  # Team or user namespace
    name: Optional[str] = None  # Optional run display name
    tags: Sequence[str] = field(default_factory=list)  # Tag list surfaced in the W&B UI
    group: Optional[str] = None  # Optional run grouping key
    notes: Optional[str] = None  # Free-form notes shown on the run page
    watch: Optional[str] = None  # Mode passed to wandb.watch (e.g. gradients, parameters)
    log_model: Optional[bool | str] = False  # Forwarded to wandb.init(log_model=...)


@dataclass
class AccelerateConfig:
    """Accelerate launcher/runner configuration for distributed training."""

    enabled: bool = False  # Mark intent to use accelerate backend
    mixed_precision: Optional[str] = None  # e.g. "fp16", "bf16"
    gradient_accumulation_steps: Optional[int] = None  # Override Trainer setting when using accelerate
    log_with: Sequence[str] = field(default_factory=list)  # Logger integrations for accelerator.log
    project_dir: Optional[str] = None  # accelerate tracker/project directory
    split_batches: bool = False  # Split large batches across processes instead of dropping
    even_batches: bool = False  # Ensure each process sees equal batch sizes


@dataclass
class DDPConfig:
    """Native torch.distributed training options."""

    enabled: bool = False  # Enable torchrun/torch.distributed backend
    backend: str = "nccl"  # torch.distributed backend (nccl, gloo)
    init_method: str = "env://"  # Initialization URL; default env var driven
    gradient_accumulation_steps: Optional[int] = None  # Override trainer accumulation for DDP loop
    find_unused_parameters: bool = False  # Pass through to DistributedDataParallel
    broadcast_buffers: bool = True  # Whether to broadcast buffers in DDP wrapper
    static_graph: bool = False  # Hint DDP graph optimization


@dataclass
class PrecisionConfig:
    fp16: bool = False
    bf16: bool = False


@dataclass
class TrainSettings:
    """High-level training configuration for readability."""

    output_dir: str
    num_epochs: float
    per_device_batch_size: int
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_steps: Optional[int] = None
    logging_steps: int = 50
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    dataloader_num_workers: int = 0


@dataclass
class EvaluationSettings:
    """Evaluation/checkpoint behavior."""

    per_device_batch_size: int
    strategy: str = "steps"
    steps: Optional[int] = None
    save_strategy: str = "steps"
    save_steps: Optional[int] = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    do_train: bool = True
    do_eval: bool = True


@dataclass
class TrainingConfig:
    """Top-level configuration bundle loaded from YAML."""

    model: ModelConfig
    dataset: DatasetConfig
    adapter: AdapterSettings
    trainer: TrainerConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    accelerate: AccelerateConfig = field(default_factory=AccelerateConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)
    train: Optional[TrainSettings] = None
    evaluation: Optional[EvaluationSettings] = None

    @staticmethod
    def load(path: str | Path) -> "TrainingConfig":
        data = yaml.safe_load(Path(path).read_text())
        return TrainingConfig.from_dict(data)

    @staticmethod
    def from_dict(config: Mapping[str, Any]) -> "TrainingConfig":
        model = ModelConfig(**config["model"])
        dataset = DatasetConfig(**config["dataset"])
        adapter = AdapterSettings(**config["adapter"])
        trainer_payload = config.get("trainer")
        train_settings = None
        evaluation_settings = None

        if "train" in config:
            precision_payload = config["train"].get("precision", {}) or {}
            train_settings = TrainSettings(
                output_dir=config["train"]["output_dir"],
                num_epochs=config["train"].get("num_epochs", config["train"].get("num_train_epochs", 1.0)),
                per_device_batch_size=config["train"].get("per_device_batch_size", config["train"].get("per_device_train_batch_size", 8)),
                gradient_accumulation_steps=config["train"].get("gradient_accumulation_steps", 1),
                warmup_ratio=config["train"].get("warmup_ratio", 0.0),
                learning_rate=config["train"].get("learning_rate", 5e-5),
                weight_decay=config["train"].get("weight_decay", 0.0),
                max_steps=config["train"].get("max_steps"),
                logging_steps=config["train"].get("logging_steps", 50),
                precision=PrecisionConfig(
                    fp16=precision_payload.get("fp16", False),
                    bf16=precision_payload.get("bf16", False),
                ),
                dataloader_num_workers=config["train"].get("dataloader_num_workers", 0),
            )

        if "evaluation" in config:
            evaluation_settings = EvaluationSettings(
                per_device_batch_size=config["evaluation"].get("per_device_batch_size", config["evaluation"].get("per_device_eval_batch_size", 8)),
                strategy=config["evaluation"].get("strategy", "steps"),
                steps=config["evaluation"].get("steps"),
                save_strategy=config["evaluation"].get("save_strategy", "steps"),
                save_steps=config["evaluation"].get("save_steps"),
                load_best_model_at_end=config["evaluation"].get("load_best_model_at_end", True),
                metric_for_best_model=config["evaluation"].get("metric_for_best_model", "accuracy"),
                greater_is_better=config["evaluation"].get("greater_is_better", True),
                do_train=config["evaluation"].get("do_train", True),
                do_eval=config["evaluation"].get("do_eval", True),
            )

        if trainer_payload is None and train_settings is not None and evaluation_settings is not None:
            trainer_payload = {
                "output_dir": train_settings.output_dir,
                "num_train_epochs": train_settings.num_epochs,
                "per_device_train_batch_size": train_settings.per_device_batch_size,
                "per_device_eval_batch_size": evaluation_settings.per_device_batch_size,
                "gradient_accumulation_steps": train_settings.gradient_accumulation_steps,
                "warmup_ratio": train_settings.warmup_ratio,
                "learning_rate": train_settings.learning_rate,
                "weight_decay": train_settings.weight_decay,
                "eval_strategy": evaluation_settings.strategy,
                "eval_steps": evaluation_settings.steps,
                "save_strategy": evaluation_settings.save_strategy,
                "save_steps": evaluation_settings.save_steps,
                "logging_steps": train_settings.logging_steps,
                "max_steps": train_settings.max_steps,
                "fp16": train_settings.precision.fp16,
                "bf16": train_settings.precision.bf16,
                "report_to": config.get("report_to", ["none"]),
                "extra": {
                    "do_train": evaluation_settings.do_train,
                    "do_eval": evaluation_settings.do_eval,
                    "load_best_model_at_end": evaluation_settings.load_best_model_at_end,
                    "metric_for_best_model": evaluation_settings.metric_for_best_model,
                    "greater_is_better": evaluation_settings.greater_is_better,
                    "dataloader_num_workers": train_settings.dataloader_num_workers,
                },
            }

        if trainer_payload is None:
            raise KeyError("Trainer configuration missing; provide either 'trainer' or both 'train' and 'evaluation' sections.")

        trainer_payload = trainer_payload.copy()
        extra = trainer_payload.pop("extra", {})
        if train_settings is not None:
            extra.setdefault("dataloader_num_workers", train_settings.dataloader_num_workers)
        trainer = TrainerConfig(extra=extra, **trainer_payload)
        runtime = RuntimeConfig(**config.get("runtime", {}))
        wandb_cfg = WandbConfig(**config.get("wandb", {})) if "wandb" in config else WandbConfig()
        accelerate_cfg = (
            AccelerateConfig(**config.get("accelerate", {})) if "accelerate" in config else AccelerateConfig()
        )
        ddp_cfg = DDPConfig(**config.get("ddp", {})) if "ddp" in config else DDPConfig()
        # Populate missing train/evaluation sections from trainer if needed
        if train_settings is None:
            train_settings = TrainSettings(
                output_dir=trainer.output_dir,
                num_epochs=trainer.num_train_epochs,
                per_device_batch_size=trainer.per_device_train_batch_size,
                gradient_accumulation_steps=trainer.gradient_accumulation_steps,
                warmup_ratio=trainer.warmup_ratio,
                learning_rate=trainer.learning_rate,
                weight_decay=trainer.weight_decay,
                max_steps=trainer.max_steps,
                logging_steps=trainer.logging_steps,
                precision=PrecisionConfig(fp16=trainer.fp16, bf16=trainer.bf16),
                dataloader_num_workers=int(trainer.extra.get("dataloader_num_workers", 0)),
            )

        if evaluation_settings is None:
            evaluation_settings = EvaluationSettings(
                per_device_batch_size=trainer.per_device_eval_batch_size,
                strategy=trainer.eval_strategy,
                steps=trainer.eval_steps,
                save_strategy=trainer.save_strategy,
                save_steps=trainer.save_steps,
                load_best_model_at_end=extra.get("load_best_model_at_end", True),
                metric_for_best_model=extra.get("metric_for_best_model", "accuracy"),
                greater_is_better=extra.get("greater_is_better", True),
                do_train=extra.get("do_train", True),
                do_eval=extra.get("do_eval", True),
            )

        return TrainingConfig(
            model=model,
            dataset=dataset,
            adapter=adapter,
            trainer=trainer,
            runtime=runtime,
            wandb=wandb_cfg,
            accelerate=accelerate_cfg,
            ddp=ddp_cfg,
            train=train_settings,
            evaluation=evaluation_settings,
        )
