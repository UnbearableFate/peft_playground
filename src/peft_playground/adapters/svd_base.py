"""Common utilities for adapters that rely on SVD of a frozen linear layer."""
from __future__ import annotations

from abc import ABC
import os
from pathlib import Path

import torch
from torch import nn

from .base import AdapterConfig, AdapterModule, SubspaceStrategy


class SVDAdapterBase(AdapterModule, ABC):
    """Shared initialisation logic for SVD-based adapters.

    Subclasses are responsible for defining the residual computation in
    `compute_delta` using the singular components stored on this base class.
    """

    def __init__(
        self,
        linear_module: nn.Linear,
        config: AdapterConfig,
        *,
        default_name: str,
        layer_name: str=None,
    ) -> None:
        config = self._normalise_config(config, default_name)
        super().__init__(linear_module, config)
        
        assert config.svd_save_path is not None, "SVDAdapterBase requires svd_save_path in config"
        assert layer_name is not None, "SVDAdapterBase requires layer_name to be specified"

        
        if os.path.exists(Path(config.svd_save_path, f"{layer_name}.pt")):
            svd_data = torch.load(Path(config.svd_save_path, f"{layer_name}.pt"), map_location="cpu")
            try:
                u, s, vh = svd_data["u"], svd_data["s"], svd_data["vh"]
                print(f"Loaded precomputed SVD for layer {layer_name}")
            except KeyError:
                raise ValueError(f"Layer name {layer_name} not found in SVD data at {config.svd_save_path}")
        else:
            with torch.no_grad():
                device = linear_module.weight.device
                weight = linear_module.weight.detach().to(device=device, dtype=torch.float32)
                try:
                    u, s, vh = torch.linalg.svd(weight, full_matrices=True)
                except RuntimeError as exc:  # pragma: no cover - defensive branch
                    raise RuntimeError(
                        f"SVD failed while initialising {self.__class__.__name__}, {exc}"
                    )
                save_path = Path(config.svd_save_path, f"{layer_name}.pt")
                if not os.path.exists(config.svd_save_path):
                    os.makedirs(config.svd_save_path)
                torch.save({"u": u.cpu(), "s": s.cpu(), "vh": vh.cpu()}, save_path)
                print(f"Computed and saved SVD for layer {layer_name} to {save_path}")
    
        if config.rank > 0:
            desired_rank = min(
                config.rank,
                linear_module.weight.shape[0],
                linear_module.weight.shape[1],
            )
            if desired_rank == 0:
                raise ValueError("SVD rank collapsed to zero; increase rank or layer size")
            active_rank = min(desired_rank, s.numel())
            u_r = u[:, :active_rank]
            s_r = s[:active_rank]
            vh_r = vh[:active_rank, :]
        else:
            u_r = u
            s_r = s
            vh_r = vh
            active_rank = s.numel()

        device = linear_module.weight.device
        dtype = linear_module.weight.dtype

        self.register_buffer("left_vectors", u_r.to(device=device, dtype=dtype))
        self.register_buffer(
            "right_vectors_t",
            vh_r.to(device=device, dtype=dtype).transpose(0, 1).contiguous(),
        )
        self.register_buffer("base_sigma", s_r.to(device=device, dtype=dtype))

        self.active_rank = active_rank
        print(
            f"Initialized {self.__class__.__name__} with rank {self.active_rank} "
        )

    @staticmethod
    def _normalise_config(config: AdapterConfig, default_name: str) -> AdapterConfig:
        return AdapterConfig(
            target_modules=config.target_modules,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            init_scale=config.init_scale,
            strategy=SubspaceStrategy.RECONSTRUCTION,
            train_bias=config.train_bias,
            name=config.name or default_name,
            extra=dict(config.extra),
            svd_save_path=config.svd_save_path,
        )

    def project_input(self, x: torch.Tensor) -> torch.Tensor:
        """Project input activations into the right-singular subspace."""
        right = self.right_vectors_t.to(dtype=x.dtype, device=x.device)
        return torch.matmul(x, right)

    def reconstruct_output(
        self,
        latent: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Reconstruct activations back into output space."""
        left = self.left_vectors.to(dtype=dtype, device=device)
        latent = latent.to(dtype=dtype, device=device)
        return torch.matmul(latent, left.transpose(0, 1))
