"""Python side of the runtime-compiled projector kernels in _cuda_kernels.cu.

The batch is packed into channel groups before each launch: a group of up to
four (forward) or eight (adjoint, backprojection) images is interleaved so one
vector load serves the whole group and the geometry is computed once for it.
Packing is a transpose of the input, a small fraction of a kernel's cost.
"""

from __future__ import annotations

import math
import os
import warnings

import torch

from ._nvrtc import KernelLibrary, runtime_available, runtime_unavailable_reason

_SOURCE = "_cuda_kernels.cu"
_FORWARD_TILE = 4  # detector bins per warp; TT_FORWARD_TW in the source
_FORWARD_WARPS = 4
_GATHER_BLOCK = (16, 8, 1)
_FAN_ADJOINT_ROWS = 2  # pixels per thread; TT_FAN_ROWS in the source
# Forward kernels are bound by image loads, so eight channels cost as much as two
# groups of four; the gathers are bound by their index arithmetic and share it.
_FORWARD_GROUP = 4
_GATHER_GROUP = 8

_library: KernelLibrary | None = None
_warned: set[str] = set()


def _kernels() -> KernelLibrary:
    global _library
    if _library is None:
        with open(os.path.join(os.path.dirname(__file__), _SOURCE), encoding="utf-8") as handle:
            _library = KernelLibrary(handle.read(), "torchtomo_projectors.cu")
    return _library


def cuda_kernels_available(device: torch.device, dtype: torch.dtype | None = None) -> bool:
    """True for float32 CUDA tensors when NVRTC and the driver load. Never raises."""
    if device.type != "cuda":
        return False
    if dtype is not None and dtype != torch.float32:
        return False
    if runtime_available():
        return True
    reason = runtime_unavailable_reason() or "unknown"
    if reason not in _warned:
        _warned.add(reason)
        warnings.warn(f"backend='cuda' falls back to the PyTorch path: {reason}", RuntimeWarning, stacklevel=3)
    return False


def _groups(batch: int, largest: int):
    start = 0
    while start < batch:
        count = min(largest, batch - start)
        width = 1 if count == 1 else 2 if count == 2 else 4 if count <= 4 else 8
        yield start, count, width
        start += count


def _pack(flat: torch.Tensor, start: int, count: int, width: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Rows start..start+count of [B, M] as [M, width], channels innermost, zero padded."""
    rows = flat[start : start + count]
    if width == 1:
        return (rows[0] * mask if mask is not None else rows[0]).contiguous()
    size = flat.shape[1]
    packed = flat.new_empty(size, width) if count == width else flat.new_zeros(size, width)
    if mask is not None:
        torch.mul(rows.t(), mask.view(-1, 1), out=packed[:, :count])
    else:
        packed[:, :count] = rows.t()
    return packed


def _flat_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    return None if mask is None else mask.reshape(-1).to(torch.float32).contiguous()


def parallel_forward(
    image: torch.Tensor, trig: torch.Tensor, mask: torch.Tensor | None, pixel_size: float
) -> torch.Tensor:
    """[B, 1, S, S] -> [B, 1, A, S]; the mask is applied to the image first."""
    batch, size = image.shape[0], image.shape[-1]
    n_angles = trig.shape[0]
    flat = image.reshape(batch, size * size)
    mask = _flat_mask(mask)
    out = image.new_empty(batch, 1, n_angles, size)
    # Samples farther than c + 2 from the centre touch only pixels outside the disc.
    c = 0.5 * (size - 1)
    r2 = (c + 2.0) ** 2 if mask is not None else 1e30
    bins = _FORWARD_TILE * _FORWARD_WARPS
    grid = (math.ceil(size / bins), n_angles, 1)
    block = (32 * _FORWARD_WARPS, 1, 1)
    library = _kernels()
    for start, count, width in _groups(batch, _FORWARD_GROUP):
        packed = _pack(flat, start, count, width, mask)
        kernel = library.function(f"parallel_forward_c{width}", image.device)
        kernel(grid, block, [packed, trig, out[start], size, n_angles, count, n_angles * size, pixel_size, r2])
    return out


def _gather_grid(size: int, rows: int = 1) -> tuple[int, int, int]:
    return (math.ceil(size / _GATHER_BLOCK[0]), math.ceil(size / (_GATHER_BLOCK[1] * rows)), 1)


def parallel_adjoint(
    sinogram: torch.Tensor, trig: torch.Tensor, mask: torch.Tensor | None, pixel_size: float
) -> torch.Tensor:
    """[B, 1, A, S] -> [B, 1, S, S], the exact transpose of parallel_forward."""
    batch, _, n_angles, size = sinogram.shape
    flat = sinogram.reshape(batch, n_angles * size)
    mask = _flat_mask(mask)
    out = sinogram.new_empty(batch, 1, size, size)
    library = _kernels()
    for start, count, width in _groups(batch, _GATHER_GROUP):
        packed = _pack(flat, start, count, width)
        kernel = library.function(f"parallel_adjoint_c{width}", sinogram.device)
        kernel(
            _gather_grid(size),
            _GATHER_BLOCK,
            [packed, trig, mask, out[start], size, n_angles, count, size * size, pixel_size],
        )
    return out


def parallel_backproject(
    sinogram: torch.Tensor, trig: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor | None, scale: float
) -> torch.Tensor:
    """FBP backprojection [B, 1, A, n_det] -> [B, 1, S, S], linear on the detector."""
    batch, _, n_angles, n_det = sinogram.shape
    size = coords.numel()
    flat = sinogram.reshape(batch, n_angles * n_det)
    mask = _flat_mask(mask)
    out = sinogram.new_empty(batch, 1, size, size)
    library = _kernels()
    for start, count, width in _groups(batch, _GATHER_GROUP):
        packed = _pack(flat, start, count, width)
        kernel = library.function(f"parallel_backproject_c{width}", sinogram.device)
        kernel(
            _gather_grid(size),
            _GATHER_BLOCK,
            [packed, trig, coords, mask, out[start], size, n_angles, n_det, count, size * size, scale],
        )
    return out


def fan_forward(
    image: torch.Tensor,
    rays: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor | None,
    n_angles: int,
    n_det: int,
    n_samples: int,
) -> torch.Tensor:
    """[B, 1, S, S] -> [B, 1, A, n_det] along the ray table."""
    batch, size = image.shape[0], image.shape[-1]
    flat = image.reshape(batch, size * size)
    mask = _flat_mask(mask)
    out = image.new_empty(batch, 1, n_angles, n_det)
    bins = _FORWARD_TILE * _FORWARD_WARPS
    grid = (math.ceil(n_det / bins), n_angles, 1)
    block = (32 * _FORWARD_WARPS, 1, 1)
    library = _kernels()
    for start, count, width in _groups(batch, _FORWARD_GROUP):
        packed = _pack(flat, start, count, width, mask)
        kernel = library.function(f"fan_forward_c{width}", image.device)
        kernel(
            grid,
            block,
            [packed, rays, weights, out[start], size, n_angles, n_det, n_samples, count, n_angles * n_det],
        )
    return out


def fan_adjoint(
    sinogram: torch.Tensor,
    rays: torch.Tensor,
    inv_steps: torch.Tensor,
    weights: torch.Tensor,
    views: torch.Tensor,
    mask: torch.Tensor | None,
    size: int,
    n_samples: int,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """[B, 1, A, n_det] -> [B, 1, S, S], the exact transpose of fan_forward."""
    batch, _, n_angles, n_det = sinogram.shape
    flat = sinogram.reshape(batch, n_angles * n_det) * weights.view(1, -1)
    mask = _flat_mask(mask)
    out = sinogram.new_empty(batch, 1, size, size)
    library = _kernels()
    for start, count, width in _groups(batch, _GATHER_GROUP):
        packed = _pack(flat, start, count, width)
        kernel = library.function(f"fan_adjoint_c{width}", sinogram.device)
        kernel(
            _gather_grid(size, _FAN_ADJOINT_ROWS),
            _GATHER_BLOCK,
            [
                packed,
                rays,
                inv_steps,
                views,
                mask,
                out[start],
                size,
                n_angles,
                n_det,
                n_samples,
                alpha,
                beta,
                count,
                size * size,
            ],
        )
    return out


def fan_backproject(
    sinogram: torch.Tensor,
    trig: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor | None,
    src: float,
    det: float,
    half_width: float,
    scale: float,
) -> torch.Tensor:
    """Weighted FBP backprojection [B, 1, A, n_det] -> [B, 1, S, S]."""
    batch, _, n_angles, n_det = sinogram.shape
    size = coords.numel()
    flat = sinogram.reshape(batch, n_angles * n_det)
    mask = _flat_mask(mask)
    out = sinogram.new_empty(batch, 1, size, size)
    library = _kernels()
    for start, count, width in _groups(batch, _GATHER_GROUP):
        packed = _pack(flat, start, count, width)
        kernel = library.function(f"fan_backproject_c{width}", sinogram.device)
        kernel(
            _gather_grid(size),
            _GATHER_BLOCK,
            [
                packed,
                trig,
                coords,
                mask,
                out[start],
                size,
                n_angles,
                n_det,
                src,
                det,
                half_width,
                count,
                size * size,
                scale,
            ],
        )
    return out
