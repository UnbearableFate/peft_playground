#!/bin/bash
# Convenience launcher for native torch.distributed DDP runs.
#
# Usage:
#   scripts/run_torchrun_ddp.sh [CONFIG_PATH]
#
# Environment overrides:
#   NNODES           Number of nodes (default: 1)
#   NPROC_PER_NODE   Processes per node (default: number of visible GPUs or 1)
#   NODE_RANK        Rank of this node (default: 0)
#   MASTER_ADDR      Master node address (default: localhost)
#   MASTER_PORT      Master port (default: 29500)

set -euo pipefail

CONFIG_PATH=${1:-configs/glue_qwen_sva_cola.yaml}
shift || true
EXTRA_ARGS=("$@")

NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
echo "[run_torchrun_ddp] Launching torch.nn.parallel DDP job"
echo "  config:        ${CONFIG_PATH}"
echo "  nodes:         ${NNODES}"
echo "  node_rank:     ${NODE_RANK}"
echo "  procs/node:    ${NPROC_PER_NODE}"
echo "  master:        ${MASTER_ADDR}:${MASTER_PORT}"

CMD=(torchrun
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  -m peft_playground.cli
  --config "${CONFIG_PATH}"
  --backend ddp)

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"
