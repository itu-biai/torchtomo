# Implementation notes and recorded measurements

[Back to the main README](../README.md). These notes preserve backend details,
migration guidance, and the project’s recorded performance experiments.
Hardware-specific timings are examples; the [benchmark repository](https://github.com/itu-biai/torchtomo-benchmark)
contains the versioned cross-library measurements and methodology.

## Fast CUDA kernels

```python
from torchtomo import FanBeam, ParallelBeam

projector = ParallelBeam(img_size=512, n_angles=360, backend="auto").cuda()
fan = FanBeam(img_size=512, n_angles=360, backend="cuda").cuda()
```

`backend="auto"` takes the CUDA kernels wherever NVRTC loads and the PyTorch path
everywhere else. It decides once, in the constructor, so `projector.backend`
reports which one it became.

`backend="cuda"` runs forward, adjoint, and FBP backprojection on CUDA kernels
written in C++ and compiled the first time they are needed by NVRTC, the
runtime compiler every CUDA build of PyTorch already installs. The kernels are
launched through the CUDA driver on PyTorch's current stream, and the compiled
binary is cached under `~/.cache/torchtomo` (`TORCHTOMO_KERNEL_CACHE` overrides
the location, an empty value disables it). There is no build step, no nvcc, and
no extra dependency. The first call compiles for about a second.

The forward and adjoint are an exact matched pair of their own: the adjoint
equals the transpose of the forward entry by entry, to float32 roundoff. They
agree with the default PyTorch path to about 1e-5. Fan beam on this backend
keeps only per-ray tables on the GPU (6 MB at 512 px and 360 angles, against
2.2 GB of sampling grids for the PyTorch path). Float64, CPU, and MPS tensors
fall back to the PyTorch path, as does a CUDA build without NVRTC.

512 x 512, batch 4, RTX 2080 Ti, milliseconds for forward / adjoint / FBP,
all measured in one process by `benchmark/benchmark_speed.py`:

| Geometry, angles | `torch` | `cuda` | `cuda`, approximate |
| --- | --- | --- | --- |
| parallel, 360 | 18.0 / 38.3 / 14.3 | 2.3 / 2.1 / 0.6 | 1.0 / 0.5 / 0.6 |
| parallel, 90 | 3.3 / 8.4 / 3.1 | 0.6 / 0.5 / 0.2 | 0.3 / 0.1 / 0.2 |
| fan, 360 | 16.2 / 53.0 / 14.9 | 2.8 / 3.2 / 1.1 | 1.3 / 0.8 / 1.1 |
| fan, 90 | 4.1 / 13.1 / 3.8 | 0.7 / 0.8 / 0.3 | 0.4 / 0.2 / 0.3 |

`backend="cuda", approximate=True` trades exactness for speed: the forward
samples through the GPU's texture units, whose 8-bit interpolation weights are
fast but not exact (about 1e-4 relative difference from the exact forward at
256 px), and the adjoint becomes a pixel-driven backprojection (about 1% from
the exact adjoint). They are no longer each other's exact transpose, so the
default stays exact.

The same table against LEAP and torch-radon is in
[torchtomo-benchmark](https://github.com/itu-biai/torchtomo-benchmark).

## Discrete adjoint and learned primal-dual

Use `projector.backward(y)` (or its equivalent `projector.adjoint(y)`) for the transpose of the
implemented forward operator, including in Learned Primal-Dual (LPD) updates.
For ordinary Euclidean tensor inner products it satisfies, up to floating-point
roundoff:

```math
\langle A x, y \rangle = \langle x, A^T y \rangle.
```

```python
import torch
from torchtomo import ParallelBeam

projector = ParallelBeam(img_size=32, n_angles=45).double()
x = torch.randn(1, 1, 32, 32, dtype=torch.float64)
y = torch.randn(1, 1, 45, 32, dtype=torch.float64, requires_grad=True)

ax = projector(x)
aty = projector.backward(y)
torch.testing.assert_close((ax * y).sum(), (x * aty).sum())

aty.square().mean().backward()  # gradients reach y and upstream dual networks
```

**Migration (0.3):** `fbp()` reconstructions change. The ramp filter is now the
DFT of Kak and Slaney's spatial kernel rather than a sampled `|f|`, which restores
the DC bin that sampling zeroes. Reconstructions no longer sit a constant below
the object (-0.017 to +0.00004 on Shepp-Logan at 512 px), and reprojecting one
returns the measurements it came from (gain 0.906 to 1.000, relative residual
0.100 to 0.005, level with LEAP). PSNR against the phantom rises 1.3 dB and SSIM
0.23 at 512 px and 360 views. Numbers from FBP runs before 0.3 are not comparable
with numbers after it.

**Migration:** `backward(y)` now computes the exact discrete adjoint. The previous
analytical backprojection is available as `backproject(y)` and is still used by
`fbp()`. It includes angular normalization and, for fan-beam geometry, distance weights.
Its detector interpolation is not the transpose of the forward image sampling,
so a global scale correction does not generally turn it into the discrete
adjoint. Existing FBP behavior is preserved. Code or trained checkpoints relying
on the old `backward()` values should use `backproject()` to preserve those values.
LPD code can continue to use `forward()`/`backward()` as a matched pair; existing
models may need retraining or step-size retuning after the operator change.

The adjoint calls `grid_sample`'s input backward kernel directly on CPU and
CUDA for parallel beam and fan beam, then uses the explicit training gradient
$g \mapsto A g$. That avoids both a throwaway forward and second derivatives of
`grid_sample`, which are unavailable in some PyTorch versions. MPS and PyTorch
builds before 1.11 (where `output_mask` was added) fall back to a temporary
forward VJP. Both paths support `torch.no_grad()` and `torch.inference_mode()`.
On MPS, bilinear sampling uses differentiable `gather` operations because some
PyTorch versions also lack the first backward derivative of `grid_sample` on
that device. Computation stays on MPS without requiring CPU fallback.
The geometry may move between calls, and it may carry a gradient; see
Differentiable Geometry below. Match projector and input device/dtype.

Default projection angles cover `[start, end)` with spacing `(end - start) / n`,
so a half-turn does not include both 0 and pi. Pass `angles=` for an explicit list.

`ParallelBeam(..., backend="triton")` (or the older `triton=True`) uses fused
Triton kernels for forward, adjoint, and FBP backprojection when Triton is
available. `backend="cuda"`, above, is faster and covers fan beam too.

`ParallelBeam(..., sparse_adjoint=True)` builds a CSR matrix of the forward map
once and applies its transpose with a sparse-dense product. Off by default: the
matrix is hundreds of megabytes at 512 px with 90 angles.

`ParallelBeam(..., grid_cache_bytes=...)` bounds how much of the per-angle
sampling grids stay resident. The default 256 MB holds a 512 px, 90 angle
forward grid (189 MB) and keeps a 512 px, 360 angle projector in the hundreds
of megabytes instead of 1.5 GB, at the cost of rebuilding about two thirds of
those grids on every call. Raise the budget above the grid size if the memory
is free; leaving the default protects LPD headroom on an 11 GB card. Set it to
0 to always rebuild.

For adjoint diagnostics, prefer float64 and report an aggregate residual as well
as per-pair relative errors: near-zero inner products can make the latter large
even for a correct adjoint. A reproducible 500-pair comparison is available with:

```bash
PYTHONPATH=src python benchmark/benchmark_adjoint.py --pairs 500 --dtype float64
```

## Differentiable geometry

The geometry is a tensor, not a constant. Every projector carries a pose table of
one row per view, `[angle, detector_shift]` for parallel beam and
`[angle, detector_shift, source_shift]` for fan beam, with the shifts lateral and
in pixels. Fan beam also carries `distances`, `[src_dist, det_dist, det_width]` in
pixels. Gradients reach all of it:

```python
import torch
from torchtomo import ParallelBeam, shepp_logan

projector = ParallelBeam(img_size=256, n_angles=180, learnable_geometry=True)
y = projector(shepp_logan(size=256)).detach()
optimizer = torch.optim.Adam(projector.parameters(), lr=0.5)

loss = (projector.forward(projector.fbp(y)) - y).pow(2).mean()
loss.backward()          # projector.pose.grad is [180, 2]
optimizer.step()
```

`learnable_geometry=True` registers the pose, and the fan's distances, as
`nn.Parameter`s, so an optimiser reaches them through `.parameters()`. Without it
they are ordinary buffers: `requires_grad_(True)` on either still gives a one-off
gradient, and `projector.set_pose(angles=..., detector_shift=...)` or
`fan.set_distances(src_dist=...)` writes the geometry with no gradient at all. A
table built as an expression in some other parameter can be assigned straight to
`projector.pose` or `fan.distances`, which is how one scalar drives every view.
Both are in the state dict, so a learnt geometry travels with its checkpoint; 0.3
checkpoints, which saved `angles` alone, still load with `strict=True`.

A constant detector shift is a centre-of-rotation error, a per-view one is
in-plane motion, and the angle column on its own is the sampling pattern.
`benchmark/calibrate_geometry.py` recovers a scanner's 3 px axis offset from the
sinogram alone, with the phantom unknown, by descending
`|| A_u fbp_u(y) - y ||^2` in the shift `u`:

| Geometry | Recovered | Error | `backend="auto"` | `backend="torch"` |
| --- | --- | --- | --- | --- |
| parallel | 2.981 px | 0.019 px | 1.3 s | 8.5 s |
| fan | 2.971 px | 0.029 px | 0.8 s | 12.1 s |

256 px, 256 views, 80 Adam steps on an RTX 2080 Ti. Adding 5% noise to the
sinogram moves the errors to 0.026 and 0.035 px. What is left is the objective's
own bias rather than the optimiser's: swept over the shift, the parallel-beam loss
has its minimum at 2.98 px at this size, and at 2.96 px at 128 px.

With `backend="cuda"` or `"auto"`, `forward()`, `adjoint()`, `backproject()` and
`fbp()` of a geometry that wants a gradient run on the CUDA kernels, with opt-in
backward kernels that return the gradient of the view table (parallel) or ray
table (fan), and of the fan's distances, and let autograd carry it to the pose.
They agree with a float64 reference to about 1e-5. On four images per batch:

| Geometry | Size, views | Pose gradient of | PyTorch path | CUDA kernels |
| --- | --- | --- | --- | --- |
| parallel | 256, 256 | forward | 71.0 ms | 0.93 ms |
| parallel | 512, 360 | forward | 401 ms | 4.62 ms |
| parallel | 512, 360 | adjoint | out of memory | 4.52 ms |
| fan | 256, 256 | forward | 108 ms | 4.06 ms |
| fan | 512, 360 | forward | out of memory | 7.32 ms |
| parallel | 256, 256 | backproject | 62.6 ms | 1.32 ms |
| parallel | 512, 360 | backproject | 365 ms | 1.82 ms |
| fan | 256, 256 | backproject | 85.7 ms | 1.68 ms |
| fan | 512, 360 | backproject | out of memory | 4.01 ms |

The backprojection's gradient is a reverse pass through its per-pixel arithmetic,
reduced per view, so it keeps no sampling grid: a few tens of MiB where the PyTorch
path's grids took gigabytes. Its sinogram gradient is the kernel's own transpose,
for a fixed geometry as well.

The kernel path is first order: a second derivative through it raises, where the
PyTorch path (any device, float64 too) differentiates the geometry to any order.
`approximate=True` differentiates the geometry through the exact PyTorch path,
since its inexact pair would give a gradient that belongs to no operator. A
shifted pose that wants no gradient, such as a scanner with a calibrated axis
offset, runs the ordinary CUDA kernels, which read the shifts from their view
table, keep their exact adjoint, and are bit for bit the 0.3.0 kernels when
nothing is shifted.
`backend="triton"` reads angles only, so a shifted pose falls back from it.
