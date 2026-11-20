import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from src.data import load_datasets
from src.model import build_model


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(42)
    torch.set_num_threads(1)

    train_ds, _, in_features = load_datasets("data/data.csv", seed=42)

    even_len = (len(train_ds) // 2) * 2
    X_even = train_ds.tensors[0][:even_len]
    y_even = train_ds.tensors[1][:even_len]
    even_ds = TensorDataset(X_even, y_even)

    local_batch_size = even_len // world_size
    sampler = DistributedSampler(
        even_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    loader = DataLoader(even_ds, batch_size=local_batch_size, sampler=sampler, shuffle=False)

    base_model, loss_fn = build_model(in_features)
    model = DDP(base_model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

    ref_path = "results/single_proc_weights.pt"
    if rank == 0:
        w = model.module.weight.detach().clone()
        b = model.module.bias.detach().clone()

        if world_size == 1:
            torch.save({"weight": w, "bias": b}, ref_path)
            print(f"[world_size=1] Reference weights saved to {ref_path}", flush=True)
        else:
            ref = torch.load(ref_path)
            max_w_diff = torch.max(torch.abs(ref["weight"] - w)).item()
            max_b_diff = torch.max(torch.abs(ref["bias"] - b)).item()
            max_diff = max(max_w_diff, max_b_diff)
            tol = 1e-5
            status = "PASS" if max_diff < tol else "FAIL"
            print(f"[world_size={world_size}] max |w1 - w2| = {max_diff:.2e} (tol={tol:.0e}) -> {status}", flush=True)
            assert max_diff < tol, f"DDP equivalence failed: {max_diff}"

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
