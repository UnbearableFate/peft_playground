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
    svd_save_path: Optional[str] = None  # Path to save SVD components
    extra: Dict[str, Any] = field(default_factory=dict)  # Free-form payload consumed by specific adapters


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
class TrainConfig:
    """Training loop hyper-parameters and runtime hints."""

    output_dir: str
    num_epochs: float = 3.0
    per_device_batch_size: int = 128
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_steps: Optional[int] = None
    logging_steps: int = 50
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    dataloader_num_workers: int = 0
    lr_scheduler_type: str = "linear"
    resume_from_checkpoint: Optional[str] = None
    report_to: Sequence[str] = field(default_factory=lambda: ["none"])
    run_name: Optional[str] = None
    push_to_hub: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.precision, Mapping):
            self.precision = PrecisionConfig(**self.precision)
        report = self.report_to
        if report is None:
            self.report_to = []
        elif isinstance(report, str):
            self.report_to = [report]
        else:
            self.report_to = list(report)
        if self.extra is None:
            self.extra = {}
        else:
            self.extra = dict(self.extra)


@dataclass
class EvaluationConfig:
    """Evaluation and checkpointing behaviour."""

    per_device_batch_size: int = 8
    strategy: str = "steps"
    steps: Optional[int] = None
    save_strategy: str = "steps"
    save_steps: Optional[int] = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    do_train: bool = True
    do_eval: bool = True
    save_total_limit: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}
        else:
            self.extra = dict(self.extra)


@dataclass
class TrainingConfig:
    """Top-level configuration bundle loaded from YAML."""

    model: ModelConfig
    dataset: DatasetConfig
    adapter: AdapterSettings
    train: TrainConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    accelerate: AccelerateConfig = field(default_factory=AccelerateConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)

    @staticmethod
    def load(path: str | Path,timestamp: str= None) -> "TrainingConfig":
        data = yaml.safe_load(Path(path).read_text())
        return TrainingConfig.from_dict(data,timestamp)

    @staticmethod
    def from_dict(config: Mapping[str, Any], timestamp: str | None) -> "TrainingConfig":
        model = ModelConfig(**config["model"])
        dataset = DatasetConfig(**config["dataset"])
        adapter = AdapterSettings(**config["adapter"])
        train_cfg = TrainConfig(**config["train"])
        evaluation_cfg = EvaluationConfig(**config["evaluation"])

        runtime = RuntimeConfig(**config.get("runtime", {}))
        wandb_cfg = WandbConfig(**config.get("wandb", {})) if "wandb" in config else WandbConfig()
        accelerate_cfg = (
            AccelerateConfig(**config.get("accelerate", {})) if "accelerate" in config else AccelerateConfig()
        )
        ddp_cfg = DDPConfig(**config.get("ddp", {})) if "ddp" in config else DDPConfig()

        experiment_name = f"{model.name_or_path.replace('/', '-')}_{dataset.name}_{dataset.subset or 'none'}_{adapter.method}"
        if train_cfg.run_name is not None:
            train_cfg.run_name += f"_{experiment_name}"
        
        train_cfg.output_dir = Path(train_cfg.output_dir, experiment_name)
        
        if timestamp:
            train_cfg.output_dir = train_cfg.output_dir / timestamp

        if wandb_cfg.enabled:
            if wandb_cfg.name is not None and wandb_cfg.name != "" and wandb_cfg.name != "None":
                print(f"Original wandb run name: {wandb_cfg.name}")
                wandb_cfg.name = experiment_name + f"_{wandb_cfg.name}"
                print(f"Updated wandb run name: {wandb_cfg.name}")
            else:
                wandb_cfg.name = experiment_name

            wandb_cfg.tags = list(wandb_cfg.tags) + [model.name_or_path.replace("/", "-"), adapter.method, dataset.name, dataset.subset or "none"]

        print(f"Outputs will be saved to {train_cfg.output_dir}, wandb run name: {wandb_cfg.name}, tags: {wandb_cfg.tags}")

        return TrainingConfig(
            model=model,
            dataset=dataset,
            adapter=adapter,
            train=train_cfg,
            evaluation=evaluation_cfg,
            runtime=runtime,
            wandb=wandb_cfg,
            accelerate=accelerate_cfg,
            ddp=ddp_cfg,
        )