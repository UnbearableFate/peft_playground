"""DoRA (Weight-Decomposed LoRA) adapter implementation."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .base import AdapterConfig, AdapterModule, SubspaceStrategy


class DoRAAdapter(AdapterModule):
    """Weight-decomposed adapter that learns magnitude and direction refinements."""

    def __init__(self, linear_module: nn.Linear, config: AdapterConfig) -> None:
        if config.rank <= 0:
            raise ValueError("DoRAAdapter requires rank > 0")

        config = AdapterConfig(
            target_modules=config.target_modules,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            init_scale=config.init_scale,
            strategy=SubspaceStrategy.COMBINATION,
            train_bias=config.train_bias,
            name=config.name or "dora",
            extra=dict(config.extra),
        )
        super().__init__(linear_module, config)

        weight = linear_module.weight.detach()
        eps = float(config.extra.get("eps", 1e-6))
        base_norm = torch.norm(weight, dim=1, keepdim=True)
        base_norm = torch.clamp(base_norm, min=eps)
        normalized_weight = weight / base_norm

        self.register_buffer("eps", torch.tensor(eps, dtype=weight.dtype))
        self.register_buffer("base_norm", base_norm)
        self.register_buffer("normalized_weight", normalized_weight)

        self.delta_g = nn.Parameter(torch.zeros_like(base_norm))

        self.down = nn.Linear(self.in_features, config.rank, bias=False)
        self.up = nn.Linear(config.rank, self.out_features, bias=False)
        self.reset_parameters()
        self._to_dtype_and_device(weight)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        if self.config.init_scale != 1.0:
            self.up.weight.data.mul_(self.config.init_scale)

    def _to_dtype_and_device(self, reference: torch.Tensor) -> None:
        dtype = reference.dtype
        device = reference.device
        self.down.to(device=device, dtype=dtype)
        self.up.to(device=device, dtype=dtype)
        self.delta_g.data = self.delta_g.data.to(device=device, dtype=dtype)
        self.base_norm.data = self.base_norm.data.to(device=device, dtype=dtype)
        self.normalized_weight.data = self.normalized_weight.data.to(device=device, dtype=dtype)
        if self.dropout_layer is not None:
            self.dropout_layer.to(device=device)

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        device = x.device

        base_norm = self.base_norm.to(device=device, dtype=dtype).view(1, -1)
        delta_g = self.delta_g.to(device=device, dtype=dtype).view(1, -1)
        normalized_weight = self.normalized_weight.to(device=device, dtype=dtype)

        delta_norm = self.up(self.down(x)) * self.scaling
        direction_term = delta_norm * base_norm

        normalized_output = F.linear(x, normalized_weight)
        magnitude_term = normalized_output * delta_g

        cross_term = delta_norm * delta_g

        return direction_term + magnitude_term + cross_term


def dora_factory(linear: nn.Linear, config: AdapterConfig) -> AdapterModule:
    return DoRAAdapter(linear, config)
