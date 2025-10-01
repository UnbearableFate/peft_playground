#!/bin/bash
current_time=$(date "+%Y%m%d%H%M")

export MASTER_ADDR=fern01
export MASTER_PORT=29400

PYTHON_BIN="/home/yu/peft_playground/.venv/bin/python"

mpirun \
 -np 2 \
 -x MASTER_ADDR -x MASTER_PORT \
 $PYTHON_BIN \
 -m peft_playground.cli \
 --config configs/glue_qwen_sva_cola.yaml