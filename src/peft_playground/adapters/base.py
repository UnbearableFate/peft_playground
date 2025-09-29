"""Adapter base classes for subspace reconstruction and extension."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn


class SubspaceStrategy(Enum):
    """High-level categorisation of how an adapter interacts with the base subspace."""

    RECONSTRUCTION = auto()
    EXTENSION = auto()
    COMBINATION = auto()


@dataclass
class AdapterConfig:
    """Shared configuration for subspace-based adapters."""

    target_modules: Sequence[str]
    rank: int
    alpha: float
    dropout: float = 0.0
    init_scale: float = 1.0
    strategy: SubspaceStrategy = SubspaceStrategy.EXTENSION
    train_bias: bool = False
    name: str | None = None
    extra: dict = field(default_factory=dict)


class AdapterModule(nn.Module):
    """Abstract module that returns the low-rank residual for a linear layer."""

    def __init__(self, linear_module: nn.Linear, config: AdapterConfig) -> None:
        super().__init__()
        if not isinstance(linear_module, nn.Linear):
            raise TypeError(
                f"AdapterModule expects nn.Linear, got {type(linear_module)}"
            )
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        self.bias = linear_module.bias if config.train_bias else None
        if self.bias is not None:
            self.bias.requires_grad = True
        self.config = config
        self.dropout_layer = nn.Dropout(config.dropout) if config.dropout > 0 else None

    @property
    def scaling(self) -> float:
        return float(self.config.alpha) / float(max(self.config.rank, 1))

    def maybe_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout_layer is None or not self.training:
            return x
        return self.dropout_layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.maybe_dropout(x)
        residual = self.compute_delta(x)
        if self.bias is not None:
            residual = residual + self.bias
        return residual

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearWithAdapter(nn.Module):
    """Wraps a frozen linear layer with a residual adapter."""

    def __init__(self, linear: nn.Linear, adapter: AdapterModule) -> None:
        super().__init__()
        self.linear = linear
        self.adapter = adapter
        # Freeze base parameters to ensure parameter-efficient fine-tuning.
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None and not adapter.config.train_bias:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        base_output = self.linear(x)
        adapter_residual = self.adapter(x)
        return base_output + adapter_residual


def iter_named_linear_modules(
    model: nn.Module, target_suffixes: Iterable[str]
) -> List[Tuple[str, nn.Linear]]:
    """Collect target linear modules that match any of the provided suffixes."""

    suffixes: Tuple[str, ...] = tuple(target_suffixes)
    matched: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not suffixes:
            matched.append((name, module))
            continue
        if any(name.endswith(suffix) for suffix in suffixes):
            matched.append((name, module))
    return matched


def replace_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a submodule on a model by its dotted path name."""

    parent = model
    path: List[str] = module_name.split(".")
    for attr in path[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, path[-1], new_module)


def attach_adapters(
    model: nn.Module,
    adapter_factory,
    adapter_config: AdapterConfig,
) -> List[Tuple[str, LinearWithAdapter]]:
    """Attach adapters to target modules; return list of (name, wrapper)."""

    wrapped_modules: List[Tuple[str, LinearWithAdapter]] = []
    for name, module in iter_named_linear_modules(model, adapter_config.target_modules):
        adapter = adapter_factory(module, adapter_config)
        wrapper = LinearWithAdapter(module, adapter)
        replace_module(model, name, wrapper)
        wrapped_modules.append((name, wrapper))
    if not wrapped_modules:
        raise ValueError(
            "No target modules were matched for adapter attachment. "
            f"Targets: {adapter_config.target_modules}"
        )
    return wrapped_modules

