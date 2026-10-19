"""Limited-angle cone beam for digital breast tomosynthesis.

The source travels on a circular arc and the flat detector stays fixed. Positions
are in the caller's length unit and use ``(x, y, z)``:

- ``+z`` points from the detector toward the source at angle 0
- ``+x`` is the tube-travel direction
- ``+y`` is parallel to the rotation axis
- detector rows increase along ``+x`` and columns along ``+y``
- the detector lies in a plane of constant ``z``

Angle 0 places the source at ``x = 0``. A positive angle is a right-handed
rotation about ``+y``, so the source moves toward ``+x``. This is the discrete
ray model used by :class:`torchtomo.conebeam.ConeBeam`: there is no FDK filter
and ``adjoint`` is the transpose of ``forward``, not an analytical backprojection.
"""

import math

import torch
from torch import nn

from .conebeam import ConeBeam


def arc_angles(
    n_angles: int,
    angular_range: float,
    first_angle: float | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Inclusive angles from the first view to the last, in radians.

    ``angular_range`` is the angle between the first and last views, not a step
    and not a half-angle. Both endpoints are acquired. A single view uses
    ``first_angle`` and ignores the range.
    """
    if int(n_angles) != n_angles or n_angles < 1:
        raise ValueError("n_angles must be a positive integer")
    if first_angle is None:
        first_angle = -angular_range / 2
    if not math.isfinite(first_angle) or not math.isfinite(angular_range):
        raise ValueError("angles must be finite")
    options = {"dtype": dtype or torch.float32, "device": device}
    if n_angles == 1:
        return torch.tensor([first_angle], **options)
    if angular_range == 0:
        raise ValueError("angular_range must be nonzero when there is more than one view")
    return torch.linspace(first_angle, first_angle + angular_range, int(n_angles), **options)


def arc_positions(
    angles: torch.Tensor,
    detector_shape: tuple[int, int],
    pixel_size: float | tuple[float, float],
    source_to_detector: float | torch.Tensor,
    source_to_axis: float | torch.Tensor,
    source_y: float | torch.Tensor = 0.0,
    detector_center: tuple[float, float] = (0.0, 0.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stationary detector panel and one source position per angle.

    ``detector_shape`` is ``(rows, columns)``. ``pixel_size`` is a scalar or
    ``(row, column)``. ``source_y`` is constant: zero puts the angle-0 source
    above ``detector_center``, while a chest-wall acquisition sets it to the
    outer edge of column 0. The detector plane is at
    ``z = -(source_to_detector - source_to_axis)``, so the perpendicular
    source-to-detector distance at angle 0 is ``source_to_detector``. Angles,
    the two distances, and ``source_y`` may be tensors that require gradients.
    """
    if angles.ndim != 1 or angles.numel() < 1:
        raise ValueError("angles must be a nonempty 1D tensor of radians")
    if angles.dtype not in (torch.float32, torch.float64) or not torch.isfinite(angles).all():
        raise ValueError("angles must be finite float32 or float64")
    rows, columns = (int(detector_shape[0]), int(detector_shape[1]))
    if rows < 1 or columns < 1 or rows != detector_shape[0] or columns != detector_shape[1]:
        raise ValueError("detector_shape must contain two positive integers, in rows, columns order")
    options = {"dtype": angles.dtype, "device": angles.device}
    source_to_detector, source_to_axis, source_y = (
        torch.as_tensor(value, **options) for value in (source_to_detector, source_to_axis, source_y)
    )
    if not all(torch.isfinite(value) for value in (source_to_detector, source_to_axis, source_y)):
        raise ValueError("source distances and source_y must be finite")
    if source_to_detector <= 0 or source_to_axis <= 0 or source_to_axis >= source_to_detector:
        raise ValueError("the rotation axis must lie between the source and the detector")
    if not all(math.isfinite(value) for value in detector_center):
        raise ValueError("detector_center must be finite")
    pitch = torch.as_tensor(pixel_size, **options).expand(2).clone()
    if not torch.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("pixel_size must be positive and finite, in row, column order")
    row_index = torch.arange(rows, **options) - (rows - 1) / 2
    column_index = torch.arange(columns, **options) - (columns - 1) / 2
    x = detector_center[0] + row_index * pitch[0]
    y = detector_center[1] + column_index * pitch[1]
    panel_xy = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1)
    detector_z = (source_to_axis - source_to_detector).expand(rows, columns, 1)
    panel = torch.cat((panel_xy, detector_z), dim=-1)
    detectors = panel.unsqueeze(0).expand(angles.shape[0], -1, -1, -1).contiguous()
    sources = torch.stack(
        (source_to_axis * angles.sin(), source_y.expand(angles.shape), source_to_axis * angles.cos()),
        dim=-1,
    )
    return sources, detectors


class BreastTomosynthesis(nn.Module):
    """Arc-source, stationary-detector cone beam and its matched transpose.

    Volumes are ``[B, 1, Z, Y, X]`` and projections are
    ``[B, 1, views, rows, columns]``. ``Y`` is the rotation-axis direction and
    ``X`` is tube travel, matching the position components. ``voxel_spacing`` is
    ``(z, y, x)`` and ``volume_center`` is ``(x, y, z)``.

    ``angles`` holds one angle per view, in radians, and ``distances`` holds
    ``(source_to_detector, source_to_axis, source_y)``. With
    ``learnable_geometry=True`` both are ``nn.Parameter`` objects. Without it
    they are buffers, and ``requires_grad_(True)`` on either still makes the
    geometry trainable. A trainable geometry rebuilds the source and detector
    positions on every call, so gradients reach it and optimiser steps take
    effect immediately. A fixed geometry keeps the positions built once.
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        *,
        angles: torch.Tensor | None = None,
        n_angles: int = 25,
        angular_range: float = math.radians(50.0),
        first_angle: float | None = None,
        detector_shape: tuple[int, int] = (128, 64),
        pixel_size: float | tuple[float, float] = 1.0,
        source_to_detector: float = 650.0,
        source_to_axis: float = 600.0,
        source_y: float = 0.0,
        detector_center: tuple[float, float] = (0.0, 0.0),
        voxel_spacing: float | tuple[float, float, float] = 1.0,
        volume_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        n_samples: int | None = None,
        ray_chunk: int = 1024,
        learnable_geometry: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if angles is None:
            angles = arc_angles(n_angles, angular_range, first_angle, dtype=dtype or torch.float32, device=device)
        else:
            angles = torch.as_tensor(angles, dtype=dtype, device=device).detach().clone()
            if angles.ndim != 1:
                raise ValueError("angles must have shape [views]")
        distances = torch.tensor(
            [float(source_to_detector), float(source_to_axis), float(source_y)],
            dtype=angles.dtype,
            device=angles.device,
        )
        self.detector_shape = (int(detector_shape[0]), int(detector_shape[1]))
        self.pixel_size = pixel_size
        self.detector_center = (float(detector_center[0]), float(detector_center[1]))
        self.learnable_geometry = learnable_geometry
        if learnable_geometry:
            self.angles = nn.Parameter(angles)
            self.distances = nn.Parameter(distances)
        else:
            self.register_buffer("angles", angles)
            self.register_buffer("distances", distances)
        with torch.no_grad():
            sources, detectors = self._positions()
        self.operator = ConeBeam(
            volume_shape,
            sources,
            detectors,
            voxel_spacing,
            volume_center,
            n_samples=n_samples,
            ray_chunk=ray_chunk,
        )

    @property
    def source_to_detector(self) -> float:
        return float(self.distances[0].detach())

    @property
    def source_to_axis(self) -> float:
        return float(self.distances[1].detach())

    @property
    def source_y(self) -> float:
        return float(self.distances[2].detach())

    def _positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        source_to_detector, source_to_axis, source_y = self.distances.unbind()
        return arc_positions(
            self.angles,
            self.detector_shape,
            self.pixel_size,
            source_to_detector,
            source_to_axis,
            source_y,
            self.detector_center,
        )

    def _geometry_is_trainable(self) -> bool:
        return self.angles.requires_grad or self.distances.requires_grad

    def _current_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._geometry_is_trainable():
            return self.operator.source_positions, self.operator.detector_positions
        sources, detectors = self._positions()
        with torch.no_grad():
            self.operator.source_positions.copy_(sources)
            self.operator.detector_positions.copy_(detectors)
        return sources, detectors

    @property
    def source_positions(self) -> torch.Tensor:
        return self._current_positions()[0]

    @property
    def detector_positions(self) -> torch.Tensor:
        return self._current_positions()[1]

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.operator._project(volume, *self._current_positions())

    def adjoint(self, projections: torch.Tensor) -> torch.Tensor:
        return self.operator._backproject(projections, *self._current_positions())

    def backward(self, projections: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`adjoint`, consistent with the other projectors."""
        return self.adjoint(projections)
