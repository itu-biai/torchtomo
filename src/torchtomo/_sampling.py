"""Bilinear sampling with zero padding and aligned corner pixels."""

import torch
import torch.nn.functional as F

# grid_sample interpolation_mode / padding_mode integer codes.
_BILINEAR = 0
_ZEROS = 0

# output_mask was added to grid_sampler_2d_backward in PyTorch 1.11 (PR 66068).
try:
    _BACKWARD_HAS_OUTPUT_MASK = "output_mask" in str(torch.ops.aten.grid_sampler_2d_backward.default._schema)
except (AttributeError, RuntimeError):
    _BACKWARD_HAS_OUTPUT_MASK = False

_INPUT_BACKWARD_SUPPORTED: dict[str, bool] = {}


def grid_sample_input_backward(
    grad_output: torch.Tensor, input_for_shape: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """Image gradient of bilinear grid_sample, grid held fixed.

    input_for_shape is read for its shape; with output_mask[1] false the kernel
    still allocates an unused grid gradient, and on CUDA it still reads input
    values for that unused path, so the tensor must be a real allocation of the
    right shape rather than a dummy view. Values do not affect the image gradient.
    """
    if _BACKWARD_HAS_OUTPUT_MASK:
        grad_input, _ = torch.ops.aten.grid_sampler_2d_backward(
            grad_output, input_for_shape, grid, _BILINEAR, _ZEROS, True, [True, False]
        )
    else:
        grad_input, _ = torch.ops.aten.grid_sampler_2d_backward(
            grad_output, input_for_shape, grid, _BILINEAR, _ZEROS, True
        )
    return grad_input


def grid_sample_input_backward_supported(device: torch.device) -> bool:
    """True when the native backward kernel runs on this device.

    MPS is excluded: some PyTorch versions have no grid_sampler_2d_backward there,
    which is why sample_bilinear already uses gather on MPS. The VJP of gather is
    the fallback adjoint on that device.
    """
    key = device.type
    cached = _INPUT_BACKWARD_SUPPORTED.get(key)
    if cached is not None:
        return cached
    if key == "mps":
        _INPUT_BACKWARD_SUPPORTED[key] = False
        return False
    try:
        probe = torch.zeros(1, 1, 2, 2, device=device)
        grid = torch.zeros(1, 2, 2, 2, device=device)
        grid_sample_input_backward(probe, probe, grid)
        _INPUT_BACKWARD_SUPPORTED[key] = True
    except (RuntimeError, NotImplementedError, TypeError):
        _INPUT_BACKWARD_SUPPORTED[key] = False
    return _INPUT_BACKWARD_SUPPORTED[key]


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
    """Differentiable bilinear sampling on CPU, CUDA, and MPS.

    A grid that wants a gradient samples through gather as well. grid_sample gives
    a first derivative in the grid but no second one (`derivative for
    aten::grid_sampler_2d_backward is not implemented`), and the adjoint is itself
    a backward pass, so differentiating it with respect to the geometry needs that
    second derivative. gather is ordinary tensor arithmetic and has every order.
    """
    if image.device.type == "mps" or (grid.requires_grad and torch.is_grad_enabled()):
        return _gather_bilinear(image, grid)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
