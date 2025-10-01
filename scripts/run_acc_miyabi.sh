#!/bin/bash
#PBS -N peft-accelerate
#PBS -q gpu              # 队列名称
#PBS -l nodes=2:ppn=1:gpus=1
#PBS -l walltime=12:00:00
#PBS -j oe

# === 1. 环境准备 ===
module load cuda/11.8
module load openmpi      # 或者集群默认的 MPI 模块
source /home/yu/miniconda3/etc/profile.d/conda.sh
conda activate py313     # 你的虚拟环境

cd "${PBS_O_WORKDIR}"

# === 2. 构造 MPI/Accelerate 需要的变量 ===
HOSTFILE="${PBS_NODEFILE}"
MASTER_ADDR=$(head -n 1 "${HOSTFILE}")
MASTER_PORT=${MASTER_PORT:-29400}
PROCS=$(sort -u "${HOSTFILE}" | wc -l)
HOSTS=$(sort -u "${HOSTFILE}" | paste -sd,)

export MASTER_ADDR MASTER_PORT

CONFIG_PATH="configs/glue_qwen_sva_cola.yaml"
PYTHON_BIN="/home/yu/miniconda3/envs/py313/bin/python"

echo "[PBS job] hosts: ${HOSTS}"
echo "[PBS job] processes: ${PROCS}"
echo "[PBS job] master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "[PBS job] config: ${CONFIG_PATH}"

# === 3. 启动 Accelerate ===
mpirun --host "${HOSTS}" \
  -np "${PROCS}" -map-by ppr:1:node \
  -x MASTER_ADDR -x MASTER_PORT -x PATH -x PYTHONPATH \
  "${PYTHON_BIN}" -m accelerate.commands.launch \
    --use_mpi \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --mixed_precision bf16 \
    --module peft_playground.cli \
    --config "${CONFIG_PATH}" \
    --backend accelerate
