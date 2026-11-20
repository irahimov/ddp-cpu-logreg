#!/usr/bin/env bash
# Run ON VM1 (ddpuser@vm1). Rank 0 runs locally, rank 1 runs on VM2 over SSH.
set -euo pipefail

cd "$(dirname "$0")/.."

MASTER_ADDR="192.168.56.101"
MASTER_PORT="29500"
IFNAME="enp0s8"
VM2="ddpuser@192.168.56.102"
REMOTE_REPO="~/ddp-cpu-logreg"

if [[ "$(hostname)" != "vm1" ]]; then
  echo "ERROR: run this script on vm1, not on $(hostname)" >&2
  exit 1
fi

mkdir -p results/logs

# Rank 1 on VM2 (background). Its log is written on VM2 and copied back at the end.
ssh -T "$VM2" "
  set -euo pipefail
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate ddp
  cd ${REMOTE_REPO} && mkdir -p results/logs
  export GLOO_SOCKET_IFNAME=${IFNAME}
  PYTHONPATH=. torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    src/train.py --config configs/config.yaml > results/logs/rank1.log 2>&1
" &
REMOTE_PID=$!

# Rank 0 on VM1 (local). Output is shown live and saved to rank0.log.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ddp
export GLOO_SOCKET_IFNAME=${IFNAME}
PYTHONPATH=. torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
  src/train.py --config configs/config.yaml 2>&1 | tee results/logs/rank0.log

# Wait for rank 1 (fails the script if rank 1 failed), then copy its log back.
wait "$REMOTE_PID"
scp -q "${VM2}:${REMOTE_REPO}/results/logs/rank1.log" results/logs/rank1.log
echo "Both nodes finished successfully!"
