# Projector performance drafts

torchtomo is 10x to 100x slower and 30x to 500x heavier than LEAP and torch-radon
(`benchmark-results/README.md`). Profiling says most of that is not the price of
staying in pure PyTorch, it is redundant work that can be removed without changing
a single output value.

Where the 29.8 ms forward goes at 512 px, 360 angles, batch 4, measured per chunk
of 16 angles across 23 chunks:

| Step | Per chunk | Share |
| --- | ---: | ---: |
| `expand(...).reshape(...)` on image and grid | 0.630 ms | 51% |
| `grid_sample` | 0.485 ms | 39% |
| `sum(dim=2)` | 0.121 ms | 10% |

More is spent copying tensors than sampling them. Separately the projector holds
1.51 GB of precomputed grids at this geometry before any call is made, and
`adjoint()` costs 105 ms, 3.5x the forward, because it differentiates a temporary
graph.

| Draft | Target | Expected | Risk |
| --- | --- | --- | --- |
| [01](01-remove-per-angle-duplication.md) | 51% of forward time | 1.5x to 2x on forward and everything built on it | low, output unchanged |
| [02](02-stop-materialising-grids.md) | 1.51 GB resident | frees the memory, likely also faster | low, output unchanged |
| [03](03-direct-discrete-adjoint.md) | 105 ms and 4.7 GB per adjoint | 2x to 3x on adjoint, order of magnitude on memory | medium, must stay exactly transpose |
| [04](04-cheap-wins.md) | launch overhead and fusion | unquantified, measure first | low |

Order matters: 01 and 02 touch the same lines and should land together. 03 is
independent and is the one that unblocks LPD batch sizes. 04 last, because it is
easier to measure once the obvious waste is gone.

Every draft keeps the operator bit-comparable. The gate for all of them is
`pytest -q` at 69 passed, `benchmark/benchmark_adjoint.py --pairs 500 --dtype
float64`, and `benchmark/test_leap_consistency.py`, which pins the forward against
an independent CUDA implementation to 0.1%.
