# TorchTomo

[![PyPI](https://img.shields.io/pypi/v/torchtomo.svg?cacheSeconds=300)](https://pypi.org/project/torchtomo/)
[![Changelog](https://img.shields.io/badge/changelog-releases-blue)](https://github.com/itu-biai/torchtomo/releases)
[![Tests](https://github.com/itu-biai/torchtomo/actions/workflows/test.yml/badge.svg)](https://github.com/itu-biai/torchtomo/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://github.com/itu-biai/torchtomo/blob/main/LICENSE)

Differentiable CT reconstruction primitives in pure PyTorch.

TorchTomo provides forward projection, an exact discrete adjoint, analytical backprojection, and filtered backprojection for parallel-beam and fan-beam geometries, with support for CPU, CUDA, and Apple Silicon (MPS).

## Features

- Pure PyTorch implementation with no custom CUDA build step
- Optional fast CUDA kernels (`backend="cuda"`, or `"auto"` to take them wherever
  they load), compiled at first use by the NVRTC that ships with PyTorch, so
  installation stays `pip install torchtomo`
- Autograd-friendly operators for learned reconstruction pipelines
- Gradients with respect to the geometry itself, not only the image: the
  per-view pose table is a tensor an optimiser can move
- Parallel-beam and fan-beam (flat detector) projectors
- Built-in FBP filters: `ramp`, `shepp-logan`, `cosine`, `hamming`, `hann`, `none`
- Built-in phantom generators for quick experiments

## Installation

```bash
pip install torchtomo
```

## Quick Start (Parallel Beam)

```python
import torch
from torchtomo import ParallelBeam, shepp_logan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

phantom = shepp_logan(size=256, device=device)  # [1, 1, 256, 256]
projector = ParallelBeam(img_size=256, n_angles=180, n_det=256).to(device)

sinogram = projector.forward(phantom)                 # [1, 1, 180, 256]
recon = projector.fbp(sinogram, filter_name="ramp")   # [1, 1, 256, 256]
```

## Fan-Beam Example

```python
from torchtomo import FanBeam, shepp_logan

phantom = shepp_logan(size=256)
projector = FanBeam(img_size=256, n_angles=360)

sinogram = projector.forward(phantom)
recon = projector.fbp(sinogram, filter_name="hann")
```

## Fast CUDA Kernels

```python
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

## Differentiable Optimization Example

```python
import torch
import torch.nn.functional as F
from torchtomo import ParallelBeam

projector = ParallelBeam(img_size=256, n_angles=180)
x = torch.zeros(1, 1, 256, 256, requires_grad=True)
y = torch.randn(1, 1, 180, 256)

loss = F.mse_loss(projector.forward(x), y)
loss.backward()  # gradients flow through projection operators
```

## Discrete Adjoint and Learned Primal-Dual

Use `projector.backward(y)` (or its equivalent `projector.adjoint(y)`) for the transpose of the
implemented forward operator, including in Learned Primal-Dual (LPD) updates.
For ordinary Euclidean tensor inner products it satisfies, up to floating-point
roundoff:

```math
\langle A x, y \rangle = \langle x, A^T y \rangle.
```

```python
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

## Differentiable Geometry

The geometry is a tensor, not a constant. Every projector carries a pose table of
one row per view, `[angle, detector_shift]` for parallel beam and
`[angle, detector_shift, source_shift]` for fan beam, with the shifts lateral and
in pixels, and gradients reach it:

```python
projector = ParallelBeam(img_size=256, n_angles=180, learnable_geometry=True)
optimizer = torch.optim.Adam(projector.parameters(), lr=0.5)

loss = (projector.forward(projector.fbp(y)) - y).pow(2).mean()
loss.backward()          # projector.pose.grad is [180, 2]
optimizer.step()
```

`learnable_geometry=True` registers the pose as an `nn.Parameter`, so an optimiser
reaches it through `.parameters()`. Without it the pose is an ordinary buffer:
`projector.pose.requires_grad_(True)` still gives a one-off gradient, and
`projector.set_pose(angles=..., detector_shift=...)` writes the geometry with no
gradient at all. A pose built as an expression in some other parameter can be
assigned straight to `projector.pose`, which is how one scalar drives every view.

A constant detector shift is a centre-of-rotation error, a per-view one is
in-plane motion, and the angle column on its own is the sampling pattern.
`benchmark/calibrate_geometry.py` recovers a scanner's 3 px axis offset from the
sinogram alone, with the phantom unknown, by descending
`|| A_u fbp_u(y) - y ||^2` in the shift `u`:

| Geometry | Recovered | Error | Time |
| --- | --- | --- | --- |
| parallel | 2.981 px | 0.019 px | 8.5 s |
| fan | 2.963 px | 0.037 px | 12.1 s |

256 px, 256 views, 80 Adam steps on an RTX 2080 Ti. Adding 5% noise to the
sinogram moves the error to 0.027 and 0.043 px. What is left is the objective's
own bias rather than the optimiser's: at 128 px and 120 views the same script
lands 0.043 px out, which is where that loss actually has its minimum.

Geometry gradients run on the PyTorch path, in float64 as well, and are
differentiable a second time; a projector with `backend="cuda"` takes that path
for the calls that want one. A shifted pose that wants no gradient, such as a
scanner with a calibrated axis offset, stays on the CUDA kernels, which read the
shifts from their pose table and keep their exact adjoint. `backend="triton"`
reads angles only, so a shifted pose falls back from it.

## API Snapshot

- `ParallelBeam(..., backend="torch" | "cuda" | "triton" | "auto")`
- `FanBeam(..., backend="torch" | "cuda" | "auto")`
- `ParallelBeam(..., learnable_geometry=True)`, `FanBeam(..., learnable_geometry=True)`
- `projector.pose`: `[n_angles, 2]` or `[n_angles, 3]`, angle then lateral shifts
- `projector.set_pose(angles=..., detector_shift=..., source_shift=...)`
- `projector.angles`, `projector.detector_shift`, `fan.source_shift`
- `projector.forward(image)`
- `projector.backward(sinogram)`: exact discrete adjoint, for LPD/iterative methods
- `projector.adjoint(sinogram)`: equivalent to `backward(sinogram)`
- `projector.backproject(sinogram)`: analytical backprojection, used by FBP
- `projector.fbp(sinogram, filter_name="ramp")`
- `apply_filter(sinogram, filter_name=...)`
- `shepp_logan(size=..., device=...)`
- `circle_phantom(size=..., n_circles=..., device=...)`
- `torchtomo.phantom.forbild(size=..., device=...)`

## Tensor Shapes

- Image: `[B, 1, H, W]`
- Sinogram: `[B, 1, n_angles, n_det]`

## Benchmarks

`benchmark/` holds torchtomo's own speed and self-consistency checks, which run
on torch alone; see [benchmark/README.md](benchmark/README.md). The library and
its tests need nothing beyond torch and numpy.

Anything that needs another library lives in
[torchtomo-benchmark](https://github.com/itu-biai/torchtomo-benchmark): the
scikit-image, LEAP, and torch-radon comparisons, and a training pipeline on ellipse phantoms and
real CT slices that scores FBP+U-Net, iRadonMAP, Learned Primal-Dual,
Noise2Inverse, and Proj2Proj against FBP, SIRT, SART, BM3D, and RED, with the
recorded results.

## Development

```bash
git clone https://github.com/itu-biai/torchtomo.git
cd torchtomo
pip install -e ".[dev]"
```

```bash
make test
make lint
make build
```

## CI/CD

- `.github/workflows/test.yml`: Python test matrix on `push` and `pull_request`
- `.github/workflows/publish.yml`: release-triggered test matrix and PyPI publish step

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).  
Commercial use by third parties requires prior written permission from the authors.
