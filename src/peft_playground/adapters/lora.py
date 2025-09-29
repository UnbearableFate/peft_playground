"""LoRA-style adapters built on the base subspace abstraction."""
from __future__ import annotations

import math
import torch
from torch import nn

from .base import AdapterConfig, AdapterModule, SubspaceStrategy


class LoRAAdapter(AdapterModule):
    """Classic low-rank adapter that performs subspace extension."""

    def __init__(self, linear_module: nn.Linear, config: AdapterConfig) -> None:
        if config.rank <= 0:
            raise ValueError("LoRAAdapter requires rank > 0")
        config = AdapterConfig(
            target_modules=config.target_modules,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            init_scale=config.init_scale,
            strategy=SubspaceStrategy.EXTENSION,
            train_bias=config.train_bias,
            name=config.name or "lora",
            extra=dict(config.extra),
        )
        super().__init__(linear_module, config)
        self.down = nn.Linear(self.in_features, config.rank, bias=False)
        self.up = nn.Linear(config.rank, self.out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        if self.config.init_scale != 1.0:
            self.up.weight.data.mul_(self.config.init_scale)

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.down(x)) * self.scaling
        return residual


class FLoRAAdapter(LoRAAdapter):
    """FLoRA introduces an intermediate transformation matrix M in the latent space."""

    def __init__(self, linear_module: nn.Linear, config: AdapterConfig) -> None:
        super().__init__(linear_module, config)
        rank = self.config.rank
        self.mid = nn.Linear(rank, rank, bias=False)
        self.reset_mid()

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def reset_mid(self) -> None:
        nn.init.eye_(self.mid.weight)
        if self.config.init_scale != 1.0:
            self.mid.weight.data.mul_(self.config.init_scale)

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.down(x)
        hidden = self.mid(hidden)
        residual = self.up(hidden) * self.scaling
        return residual


def lora_factory(linear: nn.Linear, config: AdapterConfig) -> AdapterModule:
    return LoRAAdapter(linear, config)


def flora_factory(linear: nn.Linear, config: AdapterConfig) -> AdapterModule:
    return FLoRAAdapter(linear, config)

