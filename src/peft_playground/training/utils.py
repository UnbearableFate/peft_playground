"""Utility helpers shared across training runners."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def gather_tensor_for_metrics(tensor: torch.Tensor) -> torch.Tensor:
    local = tensor.detach()
    if not is_distributed():
        return local
    world_size = dist.get_world_size()
    tensor_list = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(tensor_list, local)
    return torch.cat(tensor_list, dim=0)


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
