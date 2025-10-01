#!/bin/bash
# Launch multi-node Accelerate fine-tuning via MPI.
#
# Usage:
#   scripts/run_accelerate_mpi.sh [CONFIG_PATH]
#
# Environment overrides:
#   HOSTS             Comma-separated host list (default: fern02,fern01)
#   PROCS             Total MPI ranks / processes (default: 2)
#   PYTHON_BIN        Python executable path (default: /home/yu/miniconda3/envs/py313/bin/python)
#   MASTER_ADDR       Address of main node (default: fern02)
#   MASTER_PORT       Port for process rendezvous (default: 29400)
#   ACCELERATE_EXTRA  Extra flags passed to accelerate launch (optional)

set -euo pipefail

CONFIG_PATH=${1:-configs/glue_qwen_sva_cola.yaml}
HOSTS=${HOSTS:-fern01,fern02}
PROCS=${PROCS:-2}
PYTHON_BIN=${PYTHON_BIN:-/home/yu/peft_playground/.venv/bin/python}
MASTER_ADDR=${MASTER_ADDR:-fern01}
MASTER_PORT=${MASTER_PORT:-29400}
ACCELERATE_EXTRA=${ACCELERATE_EXTRA:-}

export MASTER_ADDR
export MASTER_PORT
export PYTHON_BIN

echo "[run_accelerate_mpi] Launching Accelerate job"
echo "  hosts:           ${HOSTS}"
echo "  processes:       ${PROCS}"
echo "  master addr:     ${MASTER_ADDR}:${MASTER_PORT}"
echo "  config:          ${CONFIG_PATH}"

mpirun --host "${HOSTS}" \
  -np "${PROCS}" -map-by ppr:1:node \
  -x MASTER_ADDR -x MASTER_PORT -x PATH -x PYTHON_BIN \
  "${PYTHON_BIN}" -m accelerate.commands.launch \
    --use_mpi \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --mixed_precision bf16 \
    ${ACCELERATE_EXTRA} \
    --module peft_playground.cli \
    --config "${CONFIG_PATH}" \
    --backend accelerate

