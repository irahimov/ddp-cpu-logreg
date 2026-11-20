# Distributed Logistic Regression on CPU (PyTorch DDP + Gloo)

Multi-node CPU distributed training using PyTorch DDP (`gloo` backend) across two Ubuntu 26.04 VMs (`192.168.56.101` and `192.168.56.102`).

## Reproducibility

1. Setup: `conda env create -f environment.yml && conda activate ddp`
2. Config: `configs/config.yaml` (`seed: 42`, `lr: 0.05`, `batch_size: 32`, `epochs: 10`)
3. Correctness Test (VM1): `PYTHONPATH=. torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29601 tests/test_ddp_equivalence.py`
4. Multi-node Train: `GLOO_SOCKET_IFNAME=enp0s8 PYTHONPATH=. torchrun --nnodes=2 --nproc_per_node=1 --node_rank=<0|1> --master_addr=192.168.56.101 --master_port=29500 src/train.py --config configs/config.yaml`
