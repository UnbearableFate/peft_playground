"""Utilities to serialize and restore adapter weights and configs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

from .base import AdapterConfig, LinearWithAdapter


def iter_attached_adapters(model: torch.nn.Module) -> Iterable[Tuple[str, LinearWithAdapter]]:
    """Yield `(name, module)` pairs for adapters attached to `model`."""
    for name, module in model.named_modules():
        if isinstance(module, LinearWithAdapter):
            yield name, module


def serialize_attached_adapters(model: torch.nn.Module) -> Dict[str, Dict[str, object]]:
    """Collect adapter configs and weights from a model."""
    serialized: Dict[str, Dict[str, object]] = {}
    for name, wrapper in iter_attached_adapters(model):
        adapter = wrapper.adapter
        serialized[name] = {
            "class": f"{adapter.__class__.__module__}.{adapter.__class__.__qualname__}",
            "config": adapter.config.to_dict(),
            "state": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
        }
    if not serialized:
        serialized["__full_model_state__"] = {
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }
    return serialized


def load_attached_adapters(
    model: torch.nn.Module,
    serialized: Dict[str, Dict[str, object]],
    *,
    strict: bool = True,
) -> Dict[str, list[str]]:
    """Load adapter states into an already-initialised model.

    Returns a summary of any adapters that were missing or incompatible.
    """
    if "__full_model_state__" in serialized:
        state = serialized["__full_model_state__"]["state"]
        model.load_state_dict(state)
        return {"missing": [], "incompatible": []}

    wrappers = {name: module for name, module in iter_attached_adapters(model)}

    missing: list[str] = []
    incompatible: list[str] = []

    for name, payload in serialized.items():
        wrapper = wrappers.get(name)
        if wrapper is None:
            missing.append(name)
            continue

        if not isinstance(payload, dict) or "state" not in payload:
            # Backwards-compatibility: payload is a plain state_dict without metadata.
            wrapper.adapter.load_state_dict(payload, strict=strict)
            continue

        try:
            stored_config = AdapterConfig.from_dict(payload.get("config", {}))
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"Warning: failed to parse adapter config for {name}: {exc}")
            if strict:
                continue
            stored_config = wrapper.adapter.config
        current_config = wrapper.adapter.config
        if stored_config.to_dict() != current_config.to_dict():
            incompatible.append(name)
            if strict:
                continue

        state_dict = payload.get("state", {})
        wrapper.adapter.load_state_dict(state_dict, strict=strict)

    return {"missing": missing, "incompatible": incompatible}


def save_adapters_to_disk(model: torch.nn.Module, output_dir: Path) -> None:
    """Persist adapters to disk in a portable format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = serialize_attached_adapters(model)
    torch.save(payload, output_dir / "adapter_state.pt")


def load_adapters_from_disk(
    model: torch.nn.Module,
    input_dir: Path,
    *,
    strict: bool = True,
) -> Dict[str, list[str]]:
    """Load adapters previously saved with `save_adapters_to_disk`."""
    payload = torch.load(input_dir / "adapter_state.pt", map_location="cpu")
    return load_attached_adapters(model, payload, strict=strict)
