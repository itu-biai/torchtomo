"""Feldkamp-Davis-Kress weights for explicit cone beam geometry.

A detector line along the filter axis and the source span a tilted fan. Changing
variables from parallel-beam FBP inside that fan weights each ray by
``|dS . nu| / D'``: ``dS`` is the source's step between views, ``nu`` the ray's
normal within the fan, and ``D'`` the distance from the source to the line. The
backprojection then weights a voxel by ``(D / l)^2``, where ``l`` is its depth along
the panel normal and ``D`` the panel's. For a circular orbit and an upright panel
these are Feldkamp's cosine weight and his ``1 / U^2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ._sampling import sample_bilinear
from .filters import FilterType, apply_filter

_FLATNESS = 1e-2
# Padded samples per apply_filter call, bounding the FFT's memory.
_FILTER_CHUNK = 1 << 25


@dataclass
class FdkGeometry:
    """weights [views, rows, columns] before the filter; matrices [views, 12] map a
    voxel index (x, y, z, 1) to (column w, row w, w); axis is the filtered dimension."""

    weights: torch.Tensor
    matrices: torch.Tensor
    axis: int


def _unit(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True)


def _dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(dim=-1)


def _panel(detectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each view's first element and row and column steps, checking the panel is a flat, even grid."""
    views, rows, columns, _ = detectors.shape
    if rows < 2 or columns < 2:
        raise ValueError("fdk needs panels of at least 2 x 2 detector elements")
    origin = detectors[:, 0, 0]
    row_step = (detectors[:, -1, 0] - origin) / (rows - 1)
    column_step = (detectors[:, 0, -1] - origin) / (columns - 1)
    row = torch.arange(rows, device=detectors.device, dtype=detectors.dtype)[:, None, None]
    column = torch.arange(columns, device=detectors.device, dtype=detectors.dtype)[:, None]
    lattice = origin[:, None, None] + row * row_step[:, None, None] + column * column_step[:, None, None]
    pitch = torch.minimum(row_step.norm(dim=-1), column_step.norm(dim=-1)).min()
    if (detectors - lattice).norm(dim=-1).amax() > _FLATNESS * pitch:
        raise ValueError("fdk needs flat panels with evenly spaced rows and columns")
    return origin, row_step, column_step


def _source_steps(sources: torch.Tensor) -> torch.Tensor:
    """Central differences along the orbit, one-sided at its ends: the arc each view stands for."""
    if sources.shape[0] < 2:
        raise ValueError("fdk needs at least two views")
    before = 2 * sources[:1] - sources[1:2]
    after = 2 * sources[-1:] - sources[-2:-1]
    padded = torch.cat([before, sources, after])
    return (padded[2:] - padded[:-2]) / 2


def _redundancy(sources: torch.Tensor, center: torch.Tensor) -> float:
    """1/2 when the orbit closes a full turn around the volume, so every line is measured twice."""
    radial = (sources - center).detach().double()
    steps = F.cosine_similarity(radial[1:], radial[:-1], dim=-1).clamp(-1, 1).acos()
    turn = steps.sum() + steps.median()
    return 0.5 if turn > 2 * math.pi - steps.median() / 2 else 1.0


def fdk_geometry(
    sources: torch.Tensor,
    detectors: torch.Tensor,
    volume_shape: tuple[int, int, int],
    spacing_xyz: torch.Tensor,
    center: torch.Tensor,
) -> FdkGeometry:
    views, rows, columns, _ = detectors.shape
    origin, row_step, column_step = _panel(detectors)
    motion = _source_steps(sources)
    along_columns = _dot(motion, _unit(column_step)).abs().sum()
    along_rows = _dot(motion, _unit(row_step)).abs().sum()
    axis = -1 if along_columns >= along_rows else -2
    filter_step = column_step if axis == -1 else row_step

    normal = _unit(torch.linalg.cross(column_step, row_step))
    normal = normal * _dot(origin - sources, normal).sign()[:, None]
    panel_depth = _dot(origin - sources, normal)
    along = _unit(filter_step)
    across = torch.linalg.cross(normal, along)
    principal = sources + panel_depth[:, None] * normal

    def coordinate(direction: torch.Tensor) -> torch.Tensor:
        start = _dot(origin - principal, direction)[:, None, None]
        row = torch.arange(rows, device=detectors.device, dtype=detectors.dtype)[:, None]
        column = torch.arange(columns, device=detectors.device, dtype=detectors.dtype)
        per_row = _dot(row_step, direction)[:, None, None]
        per_column = _dot(column_step, direction)[:, None, None]
        return start + row * per_row + column * per_column

    u, v = coordinate(along), coordinate(across)
    depth = panel_depth[:, None, None]
    line_depth = (depth.square() + v.square()).sqrt()
    motion_along = _dot(motion, along)[:, None, None]
    motion_into_line = depth * _dot(motion, normal)[:, None, None] + v * _dot(motion, across)[:, None, None]
    motion_into_line = motion_into_line / line_depth
    across_ray = (line_depth * motion_along - u * motion_into_line).abs() / (line_depth.square() + u.square()).sqrt()
    pitch = filter_step.norm(dim=-1)[:, None, None]
    weights = _redundancy(sources, center) * across_ray / (line_depth * pitch)

    steps = torch.stack([row_step, column_step], dim=1)
    gram = steps @ steps.transpose(1, 2)
    to_pixels = torch.linalg.solve(gram, steps)
    offset = _dot(to_pixels, (sources - origin)[:, None])
    inverse_depth = normal / panel_depth[:, None]
    row_w, column_w = (to_pixels + offset[..., None] * inverse_depth[:, None]).unbind(1)
    linear = torch.stack([column_w, row_w, inverse_depth], dim=1)
    constant = -_dot(linear, sources[:, None])
    size_xyz = spacing_xyz.new_tensor(volume_shape[::-1])
    first_voxel = center - (size_xyz - 1) / 2 * spacing_xyz
    by_index = linear * spacing_xyz
    constant = constant + _dot(linear, first_voxel)
    matrices = torch.cat([by_index, constant[..., None]], dim=-1).reshape(views, 12)
    return FdkGeometry(weights, matrices, axis)


def filter_lines(projections: torch.Tensor, axis: int, filter_name: FilterType) -> torch.Tensor:
    """apply_filter along one detector axis of [B, 1, views, rows, columns], a few views at a time."""
    lines = projections if axis == -1 else projections.transpose(-1, -2)
    batch, _, views, count, length = lines.shape
    padded = max(64, 1 << (2 * length - 1).bit_length())
    chunk = max(1, _FILTER_CHUNK // (batch * count * padded))
    filtered = [apply_filter(lines[:, 0, start : start + chunk], filter_name) for start in range(0, views, chunk)]
    filtered = torch.cat(filtered, dim=1)[:, None]
    return filtered if axis == -1 else filtered.transpose(-1, -2)


def backproject(filtered: torch.Tensor, matrices: torch.Tensor, volume_shape: tuple[int, int, int]) -> torch.Tensor:
    """sum over views of (D / l)^2 times the filtered projection at each voxel, bilinear, zero off the panel."""
    batch, _, views, rows, columns = filtered.shape
    options = {"device": filtered.device, "dtype": filtered.dtype}
    z, y, x = torch.meshgrid(*(torch.arange(size, **options) for size in volume_shape), indexing="ij")
    voxels = torch.stack([x, y, z, torch.ones_like(x)], dim=-1).reshape(-1, 4)
    scale = filtered.new_tensor([2 / (columns - 1), 2 / (rows - 1)])
    volume = filtered.new_zeros(batch, voxels.shape[0])
    for view in range(views):
        column_w, row_w, depth = (voxels @ matrices[view].view(3, 4).t()).unbind(-1)
        inverse = torch.where(depth > 0, 1 / depth, torch.zeros_like(depth))
        grid = torch.stack([column_w, row_w], dim=-1) * inverse[:, None] * scale - 1
        panel = filtered[:, 0, view][None]
        samples = sample_bilinear(panel, grid[None, None])[0, :, 0]
        volume = volume + samples * inverse.square()
    return volume.reshape(batch, 1, *volume_shape)
