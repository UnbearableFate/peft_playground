"""Singular Value Adjustment adapter (subspace reconstruction)."""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .base import AdapterConfig, AdapterModule, SubspaceStrategy


class SingularValueAdjustmentAdapter(AdapterModule):
    """Adjusts singular values of the frozen weight matrix along its principal subspace."""

    def __init__(self, linear_module: nn.Linear, config: AdapterConfig) -> None:
        if config.rank <= 0:
            raise ValueError("SingularValueAdjustmentAdapter requires rank > 0")

        # Force reconstruction semantics regardless of the upstream setting.
        config = AdapterConfig(
            target_modules=config.target_modules,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            init_scale=config.init_scale,
            strategy=SubspaceStrategy.RECONSTRUCTION,
            train_bias=config.train_bias,
            name=config.name or "sva",
            extra=dict(config.extra),
        )
        super().__init__(linear_module, config)

        desired_rank = min(
            config.rank,
            linear_module.weight.shape[0],
            linear_module.weight.shape[1],
        )
        if desired_rank == 0:
            raise ValueError("SVA rank is zero after clamping; increase rank or layer size")

        # Compute SVD on GPU in float32 for numerical stability.
        with torch.no_grad():
            device = linear_module.weight.device
            weight = linear_module.weight.detach().to(device=device, dtype=torch.float32)
            try:
                u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            except RuntimeError as exc:  # pragma: no cover - defensive branch
                raise RuntimeError(
                    f"SVD failed while initialising SingularValueAdjustmentAdapter, {exc}"
                )

        active_rank = min(desired_rank, s.numel())
        u_r = u[:, :active_rank]
        s_r = s[:active_rank]
        vh_r = vh[:active_rank, :]

        device = linear_module.weight.device
        dtype = linear_module.weight.dtype

        self.register_buffer("left_vectors", u_r.to(device=device, dtype=dtype))
        # Store transposed right singular vectors for efficient projection.
        self.register_buffer(
            "right_vectors_t", vh_r.to(device=device, dtype=dtype).transpose(0, 1).contiguous()
        )
        self.register_buffer("base_sigma", s_r.to(device=device, dtype=dtype))

        self.active_rank = active_rank
        self.mode: Literal["multiplicative", "additive"] = config.extra.get(
            "mode", "multiplicative"
        )
        if self.mode not in {"multiplicative", "additive"}:
            raise ValueError(
                "Unsupported SVA mode. Expected 'multiplicative' or 'additive', "
                f"got {self.mode!r}"
            )

        init_std = float(config.extra.get("init_std", 0.0))
        if init_std > 0:
            init_tensor = torch.randn(active_rank, device=device, dtype=dtype) * init_std
        else:
            init_tensor = torch.zeros(active_rank, device=device, dtype=dtype)
        self.adjustment = nn.Parameter(init_tensor)

    def _sigma_delta(self) -> torch.Tensor:
        if self.mode == "multiplicative":
            return self.base_sigma * self.adjustment
        return self.adjustment

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        right = self.right_vectors_t.to(dtype=x.dtype, device=x.device)
        left = self.left_vectors.to(dtype=x.dtype, device=x.device)
        sigma_delta = self._sigma_delta().to(dtype=x.dtype, device=x.device)

        # Project input onto the singular subspace, modulate singular values, reconstruct output.
        latent = torch.matmul(x, right)  # (batch, rank)
        latent = latent * sigma_delta  # broadcast across batch
        residual = torch.matmul(latent, left.transpose(0, 1))  # (batch, out_features)
        return residual * self.scaling


def sva_factory(linear: nn.Linear, config: AdapterConfig) -> AdapterModule:
    return SingularValueAdjustmentAdapter(linear, config)

