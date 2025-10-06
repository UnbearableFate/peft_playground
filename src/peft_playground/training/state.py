"""Shared training state dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class TrainingState:
    model: Any
    tokenizer: PreTrainedTokenizerBase
    train_dataset: Dataset
    eval_dataset: Dataset
    data_collator: Any
    metric: Any
    label_list: list[str]
    wandb_run: Any
