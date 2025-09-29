**PEFT Playground**
- Lightweight framework to study subspace reconstruction/extension PEFT derived from *See Further for Parameter Efficient Fine-tuning*.
- Core pieces: YAML-driven configs, adapter base class, LoRA/FLoRA implementations, Hugging Face training pipeline for GLUE.

**Repository Layout**
- `configs/`: ready-to-edit experiment blueprints (LoRA + FLoRA on GLUE with Qwen3-1.7B).
- `src/peft_playground/`: framework source code.
- `src/peft_playground/adapters/`: base abstractions plus LoRA/FLoRA modules.
- `src/peft_playground/config.py`: typed config loader.
- `src/peft_playground/pipeline.py`: dataset/model setup and Trainer wiring.
- `peft_document.md`: reference paper notes used for the design.

**Workflow**
- Install deps: `pip install -e .` (requires torch + transformers stack compatible with Qwen models).
- Pick a config from `configs/` or create a new one; critical fields include `adapter.method` (`lora`, `flora`, `sva`) and `adapter.target_modules` (leave empty to touch every linear layer or list suffixes such as `q_proj`, `k_proj`, `v_proj`, `o_proj`).
- Launch fine-tuning: `python -m peft_playground.cli --config configs/glue_qwen_lora.yaml`.
- Use `--dry-run` to validate the setup without training; outputs trainer arguments.
- Enable Weights & Biases logging by setting `wandb.enabled: true` in your YAML; gradients, accuracy, and GPU memory stats stream automatically.

**Extending the Framework**
- New subspace reconstruction ideas can subclass `AdapterModule` and plug into `METHOD_FACTORY`.
- Compose richer experiments by augmenting YAML: add scheduler knobs under `trainer.extra`, switch GLUE tasks by editing `dataset.subset`, or target different Hugging Face checkpoints.
- Tune observability by filling out the `wandb` section (project, run name, tags) or disabling it entirely when offline.
- Explore subspace reconstruction by setting `adapter.method: sva`, which learns singular-value adjustments on the frozen weights.

**Config Reference**
- `model`: `name_or_path` (required Hugging Face id or local path); optional `revision`, `torch_dtype` (`float32`/`float16`/`bfloat16`), `trust_remote_code`, `device_map`, `tokenizer_name`.
- `dataset`: `name` (required dataset id) plus optional `subset` (e.g. `mrpc`, `cola`), `train_split`, `eval_split`, `text_fields` (one or two columns), `label_field`, `max_length`.
- `adapter`: choose `method` (`lora`, `flora`, `sva`), list `target_modules` suffixes (empty -> all), set `rank`, `alpha`, optional `dropout`, `init_scale`, `train_bias`, `extra` metadata.
- `trainer`: `output_dir`, core hyperparams (`num_train_epochs`, per-device batch sizes, `gradient_accumulation_steps`, `warmup_ratio`, `learning_rate`, `weight_decay`), scheduling knobs (`evaluation_strategy`/`eval_strategy`, `eval_steps`, `save_strategy`, `save_steps`, `logging_steps`, `max_steps`), precision flags (`fp16`, `bf16`), reporting (`report_to`), plus `extra` passthrough to `TrainingArguments`; null values are dropped before instantiation.
- `runtime`: `seed`, `gradient_checkpointing`, `torch_compile` placeholder.
- `wandb`: toggle `enabled`, fill `project`, optional `entity`, `name`, `tags`, `group`, `notes`, `watch`, `log_model`; integration auto-adds `report_to=wandb` and closes runs safely.
