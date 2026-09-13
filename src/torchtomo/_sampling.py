"""Bilinear sampling with zero padding and aligned corner pixels."""

import torch
import torch.nn.functional as F


def _gather_bilinear(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Sample using gather, whose image gradient does not need grid_sample backward.

    Used on MPS, where some PyTorch versions lack grid_sampler_2d_backward.
    Projector grids are fixed; coordinates and weights match align_corners=True.
    """
    batch, channels, height, width = image.shape
    x = (grid[..., 0] + 1) * (width - 1) / 2
    y = (grid[..., 1] + 1) * (height - 1) / 2
    x0, y0 = x.floor().long(), y.floor().long()
    dx, dy = x - x0.to(x.dtype), y - y0.to(y.dtype)
    flat_image = image.reshape(batch, channels, -1)
    output = image.new_zeros(batch, channels, grid.shape[1] * grid.shape[2])
    for ix, iy, weight in (
        (x0, y0, (1 - dx) * (1 - dy)),
        (x0 + 1, y0, dx * (1 - dy)),
        (x0, y0 + 1, (1 - dx) * dy),
        (x0 + 1, y0 + 1, dx * dy),
    ):
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        index = iy.clamp(0, height - 1) * width + ix.clamp(0, width - 1)
        index = index.reshape(batch, 1, -1).expand(-1, channels, -1)
        values = flat_image.gather(2, index)
        output = output + values * (weight * valid).reshape(batch, 1, -1)
    return output.reshape(batch, channels, grid.shape[1], grid.shape[2])


def sample_bilinear(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Differentiable bilinear sampling on CPU, CUDA, and MPS."""
    if image.device.type == "mps":
        return _gather_bilinear(image, grid)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
