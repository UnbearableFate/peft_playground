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
class TrainingConfig:
    """Top-level configuration bundle loaded from YAML."""

    model: ModelConfig
    dataset: DatasetConfig
    adapter: AdapterSettings
    trainer: TrainerConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @staticmethod
    def load(path: str | Path) -> "TrainingConfig":
        data = yaml.safe_load(Path(path).read_text())
        return TrainingConfig.from_dict(data)

    @staticmethod
    def from_dict(config: Mapping[str, Any]) -> "TrainingConfig":
        model = ModelConfig(**config["model"])
        dataset = DatasetConfig(**config["dataset"])
        adapter = AdapterSettings(**config["adapter"])
        trainer_payload = config["trainer"].copy()
        extra = trainer_payload.pop("extra", {})
        trainer = TrainerConfig(extra=extra, **trainer_payload)
        runtime = RuntimeConfig(**config.get("runtime", {}))
        wandb_cfg = WandbConfig(**config.get("wandb", {})) if "wandb" in config else WandbConfig()
        return TrainingConfig(model=model, dataset=dataset, adapter=adapter, trainer=trainer, runtime=runtime, wandb=wandb_cfg)
