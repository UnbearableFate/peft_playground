from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch


def format_tensor_summary(tensor: torch.Tensor) -> str:
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype)
    device = str(tensor.device)
    return f"Tensor(shape={shape}, dtype={dtype}, device={device})"


def print_adapter_block(name: str, payload: Dict[str, Any]) -> None:
    adapter_class = payload.get("class", "<unknown>")
    config = payload.get("config")
    print(f"  adapter: {name}")
    print(f"    class: {adapter_class}")
    if isinstance(config, dict):
        print("    config:")
        for key, value in sorted(config.items()):
            print(f"      {key}: {value}")
    state = payload.get("state", {})
    if isinstance(state, dict):
        print("    parameters:")
        for param_name, tensor in state.items():
            if isinstance(tensor, torch.Tensor):
                summary = format_tensor_summary(tensor)
            else:
                summary = f"<non-tensor: {type(tensor).__name__}>"
            print(f"      {param_name}: {summary}")
    else:
        print(f"    raw_state_type: {type(state).__name__}")


def inspect_checkpoint(path: Path) -> None:
    checkpoint_path = path / "training_state.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    elif path.is_file():
        checkpoint_path = path
        checkpoint = torch.load(path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Could not find checkpoint file at {path}")

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Top-level keys: {list(checkpoint.keys())}")

    adapters = checkpoint.get("adapters")
    if adapters is None:
        print("No 'adapters' key present; checkpoint may be legacy full-model format.")
    elif "__full_model_state__" in adapters:
        state = adapters["__full_model_state__"].get("state", {})
        print("Adapters stored as full model state (no adapters detected).")
        if isinstance(state, dict):
            print(f"  number of tensors: {len(state)}")
        else:
            print(f"  unexpected state type: {type(state).__name__}")
    else:
        print(f"Adapters found: {list(adapters.keys())}")
        for name, payload in adapters.items():
            print_adapter_block(name, payload)

    optimizer = checkpoint.get("optimizer")
    if optimizer is not None and hasattr(optimizer, "state_dict"):
        optimizer = optimizer.state_dict()
    if isinstance(optimizer, dict):
        param_groups = optimizer.get("param_groups", [])
        print(f"Optimizer param groups: {len(param_groups)}")
        print(f"Optimizer state entries: {len(optimizer.get('state', {}))}")

    lr_scheduler = checkpoint.get("lr_scheduler")
    if isinstance(lr_scheduler, dict):
        print("LR scheduler keys:", list(lr_scheduler.keys()))

    for key in ("completed_steps", "starting_epoch", "best_metric_val"):
        if key in checkpoint:
            print(f"{key}: {checkpoint[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PEFT adapter checkpoints")
    parser.add_argument("--path", type=Path, help="Directory containing training_state.pt or the file itself")
    args = parser.parse_args()

    inspect_checkpoint(args.path)


if __name__ == "__main__":
    main()
