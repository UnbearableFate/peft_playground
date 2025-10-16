"""Singular Value Adjustment adapters built on the shared SVD base."""
from __future__ import annotations

import torch
from torch import nn

from .base import AdapterConfig, AdapterModule
from .svd_base import SVDAdapterBase


def _init_adjustment_parameter(
    module: SVDAdapterBase, config: AdapterConfig
) -> nn.Parameter:
    dtype = module.base_sigma.dtype
    device = module.base_sigma.device
    init_std = float(config.extra.get("init_std", 0.0))
    if init_std > 0:
        init_tensor = torch.randn(module.active_rank, device=device, dtype=dtype) * init_std
    else:
        init_tensor = torch.zeros(module.active_rank, device=device, dtype=dtype)
    return nn.Parameter(init_tensor)


class MultiplicativeSingularValueAdjustmentAdapter(SVDAdapterBase):
    """Residual adapter that scales base singular values element-wise."""

    def __init__(self, linear_module: nn.Linear, config: AdapterConfig, layer_name: str=None) -> None:
        super().__init__(linear_module, config, default_name="sva", layer_name=layer_name)
        self.adjustment = _init_adjustment_parameter(self, self.config)

    def _sigma_delta(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.base_sigma.to(dtype=x.dtype, device=x.device)
        adjustment = self.adjustment.to(dtype=x.dtype, device=x.device)
        return sigma * adjustment

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.project_input(x)
        latent = latent * self._sigma_delta(x)
        residual = self.reconstruct_output(latent, dtype=x.dtype, device=x.device)
        return residual * self.scaling


class AdditiveSingularValueAdjustmentAdapter(SVDAdapterBase):
    """Residual adapter that adds an offset to base singular values."""

    def __init__(self, linear_module: nn.Linear, config: AdapterConfig, layer_name: str=None) -> None:
        super().__init__(linear_module, config, default_name="sva", layer_name=layer_name)
        self.adjustment = _init_adjustment_parameter(self, self.config)

    def _sigma_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.adjustment.to(dtype=x.dtype, device=x.device)

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.project_input(x)
        latent = latent * self._sigma_delta(x)
        residual = self.reconstruct_output(latent, dtype=x.dtype, device=x.device)
        return residual * self.scaling

class SingularValueAdjustmentAdapter(MultiplicativeSingularValueAdjustmentAdapter):
    """Backward compatible alias for the multiplicative variant."""


def sva_factory(linear: nn.Linear, config: AdapterConfig, layer_name: str=None) -> AdapterModule:
    mode = config.extra.get("mode", "multiplicative")
    if mode == "multiplicative":
        return MultiplicativeSingularValueAdjustmentAdapter(linear, config, layer_name)
    if mode == "additive":
        return AdditiveSingularValueAdjustmentAdapter(linear, config, layer_name)
    raise ValueError(
        "Unsupported SVA mode. Expected 'multiplicative' or 'additive', "
        f"got {mode!r}"
    )
