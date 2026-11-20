# Scaling summary

Source runs (same configuration, 10 epochs):
- 1 rank: `rank1_single_node.csv` (VM1 only)
- 2 ranks: `rank2_multi_node.csv` (VM1 + VM2)

| Metric | 1 rank (VM1) | 2 ranks (VM1+VM2) |
|---|---|---|
| Mean epoch time | 13.39 ms | 16 779.40 ms |
| Median epoch time | 12.89 ms | 40.15 ms |
| Total 10-epoch time | 133.90 ms | 167 794.02 ms |
| Speedup (median-based, 1 rank / 2 ranks) | 1.00x | 0.32x |

The 2-rank mean is dominated by epoch 8, which took 167.0 s. Excluding that
epoch, the 2-rank mean is 86.36 ms. The median is the more representative figure.

Interpretation: on this small dataset, per-epoch communication overhead
(Gloo all-reduce over VirtualBox Host-only networking) outweighs the computation
saved by splitting the data. Two ranks are therefore slower than one, which is
expected for a 569-sample problem. The large stalls in multi-node runs are not
explained by the data; the cause is not confirmed and is reported as an
observation.

Note: with DistributedSampler, each rank processes half the data per epoch, so
the effective global batch size is 64 instead of 32. The 2-rank run therefore
performs fewer optimizer steps per epoch, and its loss values differ from the
1-rank run.
