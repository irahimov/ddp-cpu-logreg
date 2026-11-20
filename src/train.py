import argparse
import csv
import os
import socket
import time
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data import load_datasets
from src.model import build_model


def evaluate(model, test_ds, loss_fn):
    model.eval()
    X_test, y_test = test_ds.tensors
    with torch.no_grad():
        logits = model(X_test)
        loss = loss_fn(logits, y_test).item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        acc = (preds == y_test).float().mean().item()
    return loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--metrics-out", type=str, default=None, help="CSV path for per-epoch timing")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = socket.gethostname()

    print(f"rank={rank} world_size={world_size} host={hostname}", flush=True)

    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)

    train_ds, test_ds, in_features = load_datasets(cfg["data_path"], seed=cfg["seed"])
    base_model, loss_fn = build_model(in_features)
    opt = torch.optim.SGD(base_model.parameters(), lr=cfg["lr"])

    start_epoch = 1
    ckpt_path = cfg["checkpoint_path"]

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            base_model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
        else:
            base_model.load_state_dict(ckpt)
        if rank == 0:
            print(f"[Resume] Loaded checkpoint from {ckpt_path}, continuing from epoch {start_epoch}", flush=True)

    model = DDP(base_model)

    sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg["seed"],
    )
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler)

    if rank == 0:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
        if args.metrics_out:
            os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

    epoch_records = []
    total_t0 = time.perf_counter()

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        ep_t0 = time.perf_counter()
        model.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        dist.barrier()
        ep_elapsed = time.perf_counter() - ep_t0

        if rank == 0:
            test_loss, test_acc = evaluate(model.module, test_ds, loss_fn)
            train_loss = epoch_loss / len(loader)
            log_line = (
                f"Epoch {epoch:02d}/{cfg['epochs']:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc * 100:.2f}% | "
                f"Time: {ep_elapsed * 1000:.2f} ms"
            )
            print(log_line, flush=True)
            with open(cfg["log_path"], "a") as lf:
                lf.write(log_line + "\n")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                },
                ckpt_path,
            )
            epoch_records.append((epoch, train_loss, test_loss, test_acc, ep_elapsed))

    dist.barrier()
    if rank == 0:
        total_elapsed = time.perf_counter() - total_t0
        if args.metrics_out and epoch_records:
            with open(args.metrics_out, "w", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(["epoch", "train_loss", "test_loss", "test_acc", "epoch_time_sec"])
                writer.writerows(epoch_records)
        print(f"[DDP Done] Total: {total_elapsed:.4f}s | Saved: {ckpt_path}", flush=True)
    else:
        print(f"[Rank {rank} Done] Finished cleanly on {hostname}", flush=True)

    if rank == 0:
        time.sleep(1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
