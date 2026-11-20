#!/usr/bin/env bash
set -euo pipefail

MASTER_ADDR="192.168.56.101"
MASTER_PORT="29500"
IFNAME="enp0s8"

mkdir -p results/logs results/checkpoints

ssh ddpuser@192.168.56.102 "
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate ddp && \
  cd ~/ddp-cpu-logreg && mkdir -p results/logs && \
  export GLOO_SOCKET_IFNAME=${IFNAME} && \
  PYTHONPATH=. torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    src/train.py --config configs/config.yaml 2>&1 | tee results/logs/rank1.log
" &
REMOTE_PID=$!

source ~/miniconda3/etc/profile.d/conda.sh && conda activate ddp
export GLOO_SOCKET_IFNAME=${IFNAME}
PYTHONPATH=. torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
  src/train.py --config configs/config.yaml 2>&1 | tee results/logs/rank0.log

wait $REMOTE_PID
echo "Both nodes finished successfully!"
