# TorchTomo

[![PyPI](https://img.shields.io/pypi/v/torchtomo.svg)](https://pypi.org/project/torchtomo/)
[![Changelog](https://img.shields.io/github/v/release/itu-biai/torchtomo?include_prereleases&label=changelog)](https://github.com/itu-biai/torchtomo/releases)
[![Tests](https://github.com/itu-biai/torchtomo/actions/workflows/test.yml/badge.svg)](https://github.com/itu-biai/torchtomo/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/itu-biai/torchtomo/blob/main/LICENSE)

Differentiable CT reconstruction primitives in pure PyTorch.

TorchTomo provides forward projection, adjoint backprojection, and filtered backprojection for parallel-beam and fan-beam geometries, with support for CPU, CUDA, and Apple Silicon (MPS).

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

## API Snapshot

- `ParallelBeam(...)`
- `FanBeam(...)`
- `projector.forward(image)`
- `projector.backward(sinogram)`
- `projector.fbp(sinogram, filter_name="ramp")`
- `apply_filter(sinogram, filter_name=...)`
- `shepp_logan(size=..., device=...)`
- `circle_phantom(size=..., n_circles=..., device=...)`
- `torchtomo.phantom.forbild(size=..., device=...)`

## Tensor Shapes

- Image: `[B, 1, H, W]`
- Sinogram: `[B, 1, n_angles, n_det]`

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

MIT
