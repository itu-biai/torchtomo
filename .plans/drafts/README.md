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

## Outcome

Drafts 01 and 02 landed. Draft 03 was implemented, verified exact, and **reverted
on measurement**. Draft 04's chunk sweep was tried and rejected; its fusion item is
untested because the grid rewrite removed the need.

Measured at 512 px, 360 angles, batch 4, against LEAP and torch-radon on the same
card:

| Operation | before | after | gain | LEAP | torch-radon |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward | 29.94 ms | 17.88 ms | 1.67x | 2.60 ms | 0.67 ms |
| backproject | 76.83 ms | 54.49 ms | 1.41x | 1.66 ms | 0.63 ms |
| fbp | 23.58 ms | 14.28 ms | 1.65x | 7.66 ms | 0.88 ms |

| Memory | before | after | gain |
| --- | ---: | ---: | ---: |
| adjoint peak | 4709 MB | 615 MB | 7.7x |
| forward peak | 342 MB | 175 MB | 2.0x |
| resident projector | 1511 MB | 272 MB | 5.6x |

Smaller geometries gain more on time: 2.28x forward at 512 px with 90 angles,
2.44x at 256 px with 90 angles and batch 8, 3.17x on FBP there.

The headline consequence is that **LPD at 512 px with 10 iterations and width 32
now trains at batch 5**, peaking at 9.30 GB, where it previously ran out of memory
above batch 2. `--micro-batch` is no longer required for the handover's command.

I projected 2x to 3x on time and got 1.4x to 2.4x. The shortfall is draft 03,
which I expected to contribute and which instead had to be reverted.

Order matters: 01 and 02 touch the same lines and should land together. 03 is
independent and is the one that unblocks LPD batch sizes. 04 last, because it is
easier to measure once the obvious waste is gone.

Every draft keeps the operator bit-comparable. The gate for all of them is
`pytest -q` at 69 passed, `benchmark/benchmark_adjoint.py --pairs 500 --dtype
float64`, and `benchmark/test_leap_consistency.py`, which pins the forward against
an independent CUDA implementation to 0.1%.
