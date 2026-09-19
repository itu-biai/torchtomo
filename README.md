# TorchTomo

[![PyPI](https://img.shields.io/pypi/v/torchtomo.svg?cacheSeconds=300)](https://pypi.org/project/torchtomo/)
[![Changelog](https://img.shields.io/badge/changelog-releases-blue)](https://github.com/itu-biai/torchtomo/releases)
[![Tests](https://github.com/itu-biai/torchtomo/actions/workflows/test.yml/badge.svg)](https://github.com/itu-biai/torchtomo/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://github.com/itu-biai/torchtomo/blob/main/LICENSE)

Differentiable CT reconstruction primitives in pure PyTorch.

TorchTomo provides forward projection, an exact discrete adjoint, analytical backprojection, and filtered backprojection for parallel-beam and fan-beam geometries, with support for CPU, CUDA, and Apple Silicon (MPS).

## Features

- Pure PyTorch implementation with no custom CUDA build step
- Autograd-friendly operators for learned reconstruction pipelines
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
projector = FanBeam(
    img_size=256,
    n_angles=360,
    n_det=400,
    src_dist=500.0,
    det_dist=500.0,
)

sinogram = projector.forward(phantom)
recon = projector.fbp(sinogram, filter_name="hann")
```

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
CUDA for parallel beam, then uses the explicit training gradient $g \mapsto A g$.
That avoids both a throwaway forward and second derivatives of `grid_sample`,
which are unavailable in some PyTorch versions. Fan-beam, MPS, and PyTorch
builds before 1.11 (where `output_mask` was added) fall back to a temporary
forward VJP. Both paths support `torch.no_grad()` and `torch.inference_mode()`.
On MPS, bilinear sampling uses differentiable `gather` operations because some
PyTorch versions also lack the first backward derivative of `grid_sample` on
that device. Computation stays on MPS without requiring CPU fallback.
Geometry is fixed and must not change between evaluation and backpropagation;
geometry gradients are not supported. Match projector and input device/dtype.

Default projection angles cover `[start, end)` with spacing `(end - start) / n`,
so a half-turn does not include both 0 and pi. Pass `angles=` for an explicit list.

`ParallelBeam(..., triton=True)` uses fused CUDA kernels for forward, adjoint, and
FBP backprojection when Triton is available. Off by default; CPU, MPS, and a
missing Triton install keep the eager kernels.

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

## API Snapshot

- `ParallelBeam(...)`
- `FanBeam(...)`
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

## Training Example

The [ellipse reconstruction example](examples/ellipses/README.md) generates 100
phantoms with a 60/20/20 train/validation/test split, or loads real CT slices,
calibrates transmission Poisson noise to approximately 23 dB FBP PSNR, and trains
FBP+U-Net, iRadonMAP, and Learned Primal-Dual models, plus Noise2Inverse and
Proj2Proj, which train without any clean image. It scores them against FBP, SIRT,
SART, BM3D, and RED, the last of which reuses the trained U-Net as its denoiser. It saves Python
training logs, curves, checkpoints, and PNG comparisons using the existing
development dependencies.

```bash
PYTHONPATH=src .venv/bin/python examples/ellipses/train.py
```

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
