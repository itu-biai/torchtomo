"""Python side of the runtime-compiled cone beam kernels in _cone_kernels.cu.

The kernels read volumes in their own layout: a border of one zero voxel on every
side, in bricks of 4 x 4 x 4 voxels. The forward and the geometry gradient read up
to four volumes of the batch interleaved as channels, so one vector load serves the
group and every ray is set up once for it; the adjoint scatters one volume at a
time. Converting to and from the layout is one copy of the volume.
"""

from __future__ import annotations

import ctypes
import math
import os

import torch

from ._cuda_kernels import _groups, _pack
from ._nvrtc import KernelLibrary

_SOURCE = "_cone_kernels.cu"
_BLOCK = (8, 16, 1)
_GROUP = 4
_BRICK = 4

_library: KernelLibrary | None = None


def _kernels() -> KernelLibrary:
    global _library
    if _library is None:
        with open(os.path.join(os.path.dirname(__file__), _SOURCE), encoding="utf-8") as handle:
            _library = KernelLibrary(handle.read(), "torchtomo_cone.cu")
    return _library


def volume_box(volume_shape, spacing_xyz: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """The kernels' twelve floats: centre, half extent, inverse spacing, and stored centre, each (x, y, z)."""
    size_xyz = torch.tensor(volume_shape[::-1], dtype=torch.float64, device=spacing_xyz.device)
    spacing = spacing_xyz.detach().double()
    values = [center.detach().double(), size_xyz * spacing / 2, 1 / spacing, (size_xyz + 1) / 2]
    return torch.cat(values).to(torch.float32).contiguous()


def kernels_fit(volume_shape, projection_shape) -> bool:
    """The kernels index the stored volume and the projections with 32 bits."""
    return _stored_size(volume_shape) < 2**32 and math.prod(projection_shape) < 2**32


def _bricks(volume_shape) -> tuple[int, int, int]:
    return tuple(-(-(size + 2) // _BRICK) for size in volume_shape)


def _stored_size(volume_shape) -> int:
    return math.prod(_bricks(volume_shape)) * _BRICK**3


def _lattice(stored: torch.Tensor, volume_shape) -> torch.Tensor:
    """A [bricks * 64, group] stored volume as [group, Z, Y, X], the border cropped."""
    depth, height, width = volume_shape
    bz, by, bx = _bricks(volume_shape)
    b = _BRICK
    lattice = stored.view(bz, by, bx, b, b, b, -1).permute(6, 0, 3, 1, 4, 2, 5)
    return lattice.reshape(-1, b * bz, b * by, b * bx)[:, 1 : depth + 1, 1 : height + 1, 1 : width + 1]


def _brick(volume: torch.Tensor, start: int, count: int, group: int) -> torch.Tensor:
    """Volumes start..start+count of [B, 1, Z, Y, X] in the kernels' layout, [bricks * 64, group]."""
    depth, height, width = volume.shape[2:]
    bz, by, bx = _bricks((depth, height, width))
    b = _BRICK
    padded = volume.new_zeros(count, b * bz, b * by, b * bx)
    padded[:, 1 : depth + 1, 1 : height + 1, 1 : width + 1] = volume[start : start + count, 0]
    bricks = padded.view(count, bz, b, by, b, bx, b).permute(1, 3, 5, 2, 4, 6, 0)
    if count == group:
        return bricks.reshape(-1, group)
    stored = volume.new_zeros(bz, by, bx, b, b, b, group)
    stored[..., :count] = bricks
    return stored.view(-1, group)


def _grid(projection_shape) -> tuple[int, int, int]:
    views, rows, columns = projection_shape
    return (math.ceil(columns / _BLOCK[0]), math.ceil(rows / _BLOCK[1]), views)


def _geometry(sources: torch.Tensor, detectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return sources.detach().float().contiguous(), detectors.detach().float().contiguous()


def cone_forward(
    volume: torch.Tensor, sources: torch.Tensor, detectors: torch.Tensor, box: torch.Tensor, n_samples: int
) -> torch.Tensor:
    """[B, 1, Z, Y, X] -> [B, 1, views, rows, columns]."""
    batch, _, depth, height, width = volume.shape
    projection_shape = tuple(detectors.shape[:3])
    plane = math.prod(projection_shape)
    sources, detectors = _geometry(sources, detectors)
    out = volume.new_empty(batch, 1, *projection_shape)
    library = _kernels()
    for start, count, group in _groups(batch, _GROUP):
        stored = _brick(volume, start, count, group)
        kernel = library.function(f"cone_forward_c{group}", volume.device)
        kernel(
            _grid(projection_shape),
            _BLOCK,
            [stored, sources, detectors, box, out[start], width, height, depth, *projection_shape[1:], n_samples]
            + [count, ctypes.c_size_t(plane)],
        )
    return out


def cone_adjoint(
    projections: torch.Tensor,
    sources: torch.Tensor,
    detectors: torch.Tensor,
    box: torch.Tensor,
    n_samples: int,
    volume_shape: tuple[int, int, int],
) -> torch.Tensor:
    """[B, 1, views, rows, columns] -> [B, 1, Z, Y, X], the transpose of cone_forward."""
    batch = projections.shape[0]
    projection_shape = tuple(projections.shape[2:])
    depth, height, width = volume_shape
    sources, detectors = _geometry(sources, detectors)
    flat = projections.reshape(batch, -1)
    out = projections.new_empty(batch, 1, *volume_shape)
    kernel = _kernels().function("cone_adjoint_c1", projections.device)
    for item in range(batch):
        stored = projections.new_zeros(_stored_size(volume_shape), 1)
        kernel(
            _grid(projection_shape),
            _BLOCK,
            [flat[item], sources, detectors, box, stored, width, height, depth, *projection_shape[1:], n_samples],
        )
        out[item, 0] = _lattice(stored, volume_shape)[0]
    return out


def cone_geometry_grad(
    volume: torch.Tensor,
    grad_projections: torch.Tensor,
    sources: torch.Tensor,
    detectors: torch.Tensor,
    box: torch.Tensor,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """d<g, A x> / d(sources) and / d(detectors), [views, 3] and [views, rows, columns, 3], summed over the batch."""
    batch, _, depth, height, width = volume.shape
    projection_shape = tuple(detectors.shape[:3])
    sources, detectors = _geometry(sources, detectors)
    grad_flat = grad_projections.reshape(batch, -1)
    grad_sources = torch.zeros_like(sources)
    grad_detectors = torch.empty_like(detectors)
    library = _kernels()
    for start, count, group in _groups(batch, _GROUP):
        stored = _brick(volume, start, count, group)
        packed_grad = _pack(grad_flat, start, count, group)
        group_detectors = grad_detectors if start == 0 else torch.empty_like(detectors)
        kernel = library.function(f"cone_geometry_grad_c{group}", volume.device)
        kernel(
            _grid(projection_shape),
            _BLOCK,
            [stored, packed_grad, sources, detectors, box, grad_sources, group_detectors, width, height, depth]
            + [*projection_shape[1:], n_samples],
        )
        if start > 0:
            grad_detectors += group_detectors
    return grad_sources, grad_detectors
