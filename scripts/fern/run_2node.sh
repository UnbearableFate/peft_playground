#!/bin/bash
current_time=$(date "+%Y%m%d%H%M")

export MASTER_ADDR=fern01
export MASTER_PORT=29400

PYTHON_BIN="/home/yu/peft_playground/.venv/bin/python"

mpirun --host fern01,fern02 \
 -np 2 -map-by ppr:1:node \
 -x MASTER_ADDR -x MASTER_PORT \
 $PYTHON_BIN \
 -m peft_playground.cli \
 --backend ddp \
 --config /home/yu/peft_playground/configs/glue_qwen_mam_cola.yaml