# TorchTomo

**Differentiable CT Reconstruction in Pure PyTorch**

A lightweight library for CT forward and back projection that works on any device (CPU, CUDA, MPS) without compilation.

## Features

- **Pure PyTorch** - No CUDA compilation, works on Mac/Linux/Windows
- **Differentiable** - Full autograd support for learned reconstruction
- **Multiple geometries** - Parallel beam, fan beam (flat detector)
- **FBP filters** - Ramp, Shepp-Logan, Cosine, Hamming, Hann
- **Test phantoms** - Shepp-Logan, circle phantom, FORBILD

## Installation

```bash
pip install torchtomo
```

For development:

```bash
git clone https://github.com/biailab/torchtomo.git
cd torchtomo
pip install -e ".[dev]"
```

## Quick Start

```python
from torchtomo import ParallelBeam, FanBeam, shepp_logan

# Create phantom
phantom = shepp_logan(size=256)  # [1, 1, 256, 256]

# Parallel beam CT
projector = ParallelBeam(img_size=256, n_angles=180, n_det=256)
sinogram = projector.forward(phantom)      # Forward projection
recon = projector.fbp(sinogram)            # Filtered back-projection

# Fan beam CT
projector = FanBeam(
    img_size=256,
    n_angles=360,
    n_det=400,
    src_dist=500,    # Source to isocenter
    det_dist=500,    # Isocenter to detector
)
sinogram = projector.forward(phantom)
recon = projector.fbp(sinogram)
```

## Learned Reconstruction

```python
import torch
import torch.nn as nn
from torchtomo import FanBeam

class LearnedReconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = FanBeam(img_size=256, n_angles=360)
        self.denoiser = UNet(1, 1)  # Your CNN
        self.step_size = nn.Parameter(torch.tensor(0.1))

    def forward(self, sinogram, n_iters=5):
        # Initial FBP
        x = self.projector.fbp(sinogram)

        # Unrolled iterations
        for _ in range(n_iters):
            # Data consistency
            residual = self.projector.forward(x) - sinogram
            grad = self.projector.backward(residual)
            x = x - self.step_size * grad

            # Learned denoising
            x = self.denoiser(x)

        return x

# Train with autograd
model = LearnedReconstruction()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for sinogram, target in dataloader:
    recon = model(sinogram)
    loss = nn.functional.mse_loss(recon, target)
    loss.backward()  # Works!
    optimizer.step()
```

## API Reference

### Parallel Beam

```python
ParallelBeam(
    img_size=256,        # Image size (square)
    n_angles=180,        # Number of projection angles
    n_det=None,          # Detector elements (default: img_size)
    angle_range=(0, pi), # Angle range in radians
    circle=True,         # Mask to inscribed circle
)
```

### Fan Beam

```python
FanBeam(
    img_size=256,           # Image size
    n_angles=360,           # Number of angles
    n_det=400,              # Detector elements
    src_dist=500.0,         # Source to isocenter distance
    det_dist=500.0,         # Isocenter to detector distance
    det_width=None,         # Detector width (default: 1.5 * img_size)
    angle_range=(0, 2*pi),  # Full rotation
    n_samples=256,          # Samples per ray
    circle=True,            # Mask to inscribed circle
)
```

### Methods

```python
projector.forward(image)              # Image -> Sinogram
projector.backward(sinogram)          # Sinogram -> Image (adjoint)
projector.fbp(sinogram, filter='ramp')  # Filtered back-projection
```

### Filters

```python
from torchtomo import apply_filter

filtered = apply_filter(sinogram, filter_name='ramp')
# Options: 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', 'none'
```

### Phantoms

```python
from torchtomo import shepp_logan, circle_phantom

phantom = shepp_logan(size=256)      # Classic test phantom
phantom = circle_phantom(size=256)   # Random circles
```

## Geometry Reference

### Parallel Beam
```
    ═══════════════════  Rays (parallel)
    ║ ║ ║ ║ ║ ║ ║ ║ ║ ║
    ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    ┌─────────────────┐
    │                 │
    │     Image       │
    │                 │
    └─────────────────┘
    ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  Detector
```

### Fan Beam
```
                    Source
                      ◯
                     /|\
                    / | \
                   /  |  \
                  /   |   \
                 /    |    \
                / ┌───┴───┐ \
               /  │ Image │  \
              /   └───────┘   \
             /        |        \
            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                 Detector
```

## Comparison with torch-radon

| Feature | TorchTomo | torch-radon |
|---------|-----------|-------------|
| Mac support | Yes | No (CUDA only) |
| GPU acceleration | MPS/CPU | CUDA (fast) |
| Differentiable | Yes | Yes |
| Installation | pip install | Compile |
| Fan beam | Yes | Yes |
| Cone beam | No | No |

**Recommendation:**
- Use **TorchTomo** for development on Mac or CPU
- Use **torch-radon** for training on CUDA (faster)

## Testing

```bash
make test
```

## License

MIT License - Free for academic and commercial use.
