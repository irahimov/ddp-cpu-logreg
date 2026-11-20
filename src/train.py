import os
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
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)

    train_ds, test_ds, in_features = load_datasets(cfg["data_path"], seed=cfg["seed"])
    base_model, loss_fn = build_model(in_features)
    model = DDP(base_model)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])

    sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg["seed"],
    )
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler)

    if rank == 0:
        os.makedirs(os.path.dirname(cfg["checkpoint_path"]), exist_ok=True)
        os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
        print(f"[DDP Start] rank={rank}/{world_size} | train_samples={len(train_ds)}", flush=True)

    start_time = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        if rank == 0:
            test_loss, test_acc = evaluate(model.module, test_ds, loss_fn)
            log_line = (
                f"Epoch {epoch:02d}/{cfg['epochs']:02d} | "
                f"Train Loss: {epoch_loss / len(loader):.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc * 100:.2f}%"
            )
            print(log_line, flush=True)
            with open(cfg["log_path"], "a") as lf:
                lf.write(log_line + "\n")

    if rank == 0:
        elapsed = time.time() - start_time
        torch.save(model.module.state_dict(), cfg["checkpoint_path"])
        print(f"[DDP Done] Elapsed: {elapsed:.3f}s | Saved: {cfg['checkpoint_path']}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
