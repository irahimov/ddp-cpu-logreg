# Distributed Logistic Regression on CPU (PyTorch DDP + Gloo)

Multi-node CPU training of a logistic regression model with PyTorch
`DistributedDataParallel` (DDP) over the Gloo backend. Training runs on two
Ubuntu 26.04 Server VMs (VirtualBox) connected through a Host-only network.

## Overview

- **Model:** Logistic regression, `torch.nn.Linear(30, 1)` with `BCEWithLogitsLoss`
- **Dataset:** Breast Cancer Wisconsin (Diagnostic), 569 samples, 30 features,
  loaded with `sklearn.datasets.load_breast_cancer` (same data as the Kaggle
  "Breast Cancer Wisconsin" dataset, originally from UCI WDBC)
- **Backend:** `gloo` (CPU only, no GPU)
- **Topology:** 2 nodes, 1 process per node, `world_size = 2`
  (VM1 = rank 0 / master, VM2 = rank 1)

## Repository structure

## Network setup

| Node | Hostname | IP (Host-only, `enp0s8`) |
|------|----------|--------------------------|
| VM1  | vm1      | 192.168.56.101 (master)  |
| VM2  | vm2      | 192.168.56.102           |

Both VMs must reach each other by IP (verify with `ping` in both directions).
`localhost` / `127.0.0.1` cannot be used as `MASTER_ADDR` for multi-node runs.

## Setup (on both VMs)

```bash
conda env create -f environment.yml
conda activate ddp
python -c "import torch, torch.distributed as dist; print(torch.__version__, dist.is_gloo_available())"
```

Expected output: a `+cpu` torch version and `True`.

## Running

### 1. DDP equivalence test (VM1)

Checks that 1-process and 2-process DDP produce the same weights.

```bash
PYTHONPATH=. torchrun --nproc_per_node=2 --master_addr=127.0.0.1 \
  --master_port=29601 tests/test_ddp_equivalence.py
```

### 2. Multi-node training (VM1 and VM2, started together)

```bash
export GLOO_SOCKET_IFNAME=enp0s8
PYTHONPATH=. torchrun --nnodes=2 --nproc_per_node=1 --node_rank=<0|1> \
  --master_addr=192.168.56.101 --master_port=29500 \
  src/train.py --config configs/config.yaml
```

Use `--node_rank=0` on VM1 and `--node_rank=1` on VM2.
To resume from the last checkpoint, add `--resume`.

## Results

- **Training:** 10 epochs, both ranks complete with exit code 0. Logs in `results/logs/`.
- **Equivalence test:** `results/tests/ddp_equivalence.log`.
- **Scaling (1 vs 2 ranks):** `results/scaling/scaling_summary.md`. On a dataset
  this small, communication overhead dominates, so 2 ranks are not faster.
- **Network bandwidth:** `results/network/iperf3_server.txt`.
- **Fault tolerance:** killing rank 1 mid-training makes rank 0 fail or hang.
  See `results/fault/rank0_fault_test.log`.

## Known limitations

- The dataset is small, so distributed training gives no speedup.
- Fault tolerance is not automatic: when a rank fails, the job stops and must be
  restarted with `--resume`.
- IP addresses and the network interface name are specific to this VirtualBox setup.
