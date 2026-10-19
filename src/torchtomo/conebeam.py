"""3D cone beam projection from explicit source and detector coordinates."""

import itertools

import torch
from torch import nn


class ConeBeam(nn.Module):
    """Ray integration and its matched transpose in physical coordinates.

    Volumes have shape ``[B, 1, Z, Y, X]`` and projections have shape
    ``[B, 1, views, detector_rows, detector_columns]``. Source positions are
    ``[views, 3]`` and detector element positions are ``[views, rows, columns, 3]``;
    coordinate components are always ``(x, y, z)``. All positions, voxel spacing,
    and the volume centre must use the same length unit. Volume values are linear
    attenuation coefficients in the inverse of that unit.

    Explicit detector coordinates support flat and curved panels, helical
    trajectories, and source offsets without approximating them as 2D slices.
    Ray samples are generated in chunks, rather than cached for the whole scan.
    This reference backend uses PyTorch on CPU or CUDA. Geometry tensors can
    require gradients, including when derived from other trainable parameters.
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        source_positions: torch.Tensor,
        detector_positions: torch.Tensor,
        voxel_spacing: float | tuple[float, float, float] = 1.0,
        volume_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        n_samples: int | None = None,
        ray_chunk: int = 1024,
    ):
        super().__init__()
        if len(volume_shape) != 3 or any(int(size) != size or size < 2 for size in volume_shape):
            raise ValueError("volume_shape must contain three integers of at least 2, in Z, Y, X order")
        if source_positions.ndim != 2 or source_positions.shape[-1] != 3 or source_positions.shape[0] < 1:
            raise ValueError("source_positions must have shape [views, 3]")
        if detector_positions.ndim != 4 or detector_positions.shape[-1] != 3:
            raise ValueError("detector_positions must have shape [views, rows, columns, 3]")
        if detector_positions.shape[0] != source_positions.shape[0] or min(detector_positions.shape[1:3]) < 1:
            raise ValueError("detector_positions must contain a nonempty panel for each source")
        if source_positions.dtype not in (torch.float32, torch.float64):
            raise ValueError("geometry must use float32 or float64")
        if detector_positions.device != source_positions.device or detector_positions.dtype != source_positions.dtype:
            raise ValueError("source and detector geometry must share a device and dtype")
        if not torch.isfinite(source_positions).all() or not torch.isfinite(detector_positions).all():
            raise ValueError("geometry must be finite")
        self.volume_shape = tuple(int(size) for size in volume_shape)
        self.projection_shape = tuple(detector_positions.shape[:3])
        self.n_samples = max(volume_shape) * 2 if n_samples is None else n_samples
        if int(self.n_samples) != self.n_samples or self.n_samples < 1 or int(ray_chunk) != ray_chunk or ray_chunk < 1:
            raise ValueError("n_samples and ray_chunk must be positive integers")
        self.n_samples = int(self.n_samples)
        self.ray_chunk = int(ray_chunk)
        options = {"device": source_positions.device, "dtype": source_positions.dtype}
        spacing = torch.as_tensor(voxel_spacing, **options).expand(3).clone()
        center = torch.as_tensor(volume_center, **options)
        if center.shape != (3,) or not torch.isfinite(center).all():
            raise ValueError("volume_center must contain three finite coordinates in X, Y, Z order")
        if not torch.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("voxel_spacing must be positive and finite, in Z, Y, X order")
        self.register_buffer("source_positions", source_positions)
        self.register_buffer("detector_positions", detector_positions)
        self.register_buffer("spacing_xyz", spacing.flip(0))
        self.register_buffer("volume_center", center)
        self.register_buffer("corners", torch.tensor(list(itertools.product((0, 1), repeat=3)), device=spacing.device))

    def _check_input(self, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
        if tensor.ndim != len(shape) + 2 or tensor.shape[1] != 1 or tuple(tensor.shape[2:]) != shape:
            raise ValueError(f"expected [B, 1, {', '.join(map(str, shape))}], got {tuple(tensor.shape)}")
        if tensor.dtype != self.source_positions.dtype or tensor.device != self.source_positions.device:
            raise ValueError("input and projector geometry must share a device and dtype")

    def _stencil(
        self, start: int, end: int, sources: torch.Tensor, detectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        panel_size = self.projection_shape[1] * self.projection_shape[2]
        ray_ids = torch.arange(start, end, device=sources.device)
        origins = sources[ray_ids // panel_size]
        directions = detectors.reshape(-1, 3)[start:end] - origins
        ray_length = directions.norm(dim=-1)
        directions = directions / ray_length.clamp_min(torch.finfo(directions.dtype).tiny)[:, None]
        origins = origins - self.volume_center
        size_xyz = self.spacing_xyz.new_tensor(self.volume_shape[::-1])
        half_extent = size_xyz * self.spacing_xyz / 2
        moving = directions.abs() > torch.finfo(directions.dtype).eps
        safe_directions = torch.where(moving, directions, torch.ones_like(directions))
        first = (-half_extent - origins) / safe_directions
        last = (half_extent - origins) / safe_directions
        inside = origins.abs() <= half_extent
        lower = torch.where(moving, torch.minimum(first, last), -torch.inf)
        upper = torch.where(moving, torch.maximum(first, last), torch.where(inside, torch.inf, -torch.inf))
        entry = lower.amax(dim=-1).clamp_min(0)
        exit = torch.minimum(upper.amin(dim=-1), ray_length)
        length = (exit - entry).clamp_min(0)
        length = torch.where(ray_length > 0, length, 0.0)
        fractions = (torch.arange(self.n_samples, device=origins.device, dtype=origins.dtype) + 0.5) / self.n_samples
        distances = entry[:, None] + length[:, None] * fractions
        points = origins[:, None, :] + distances[..., None] * directions[:, None, :]
        coordinates = points / self.spacing_xyz + (size_xyz - 1) / 2
        base = coordinates.floor().long()
        fraction = coordinates - base.to(coordinates.dtype)
        indices = base[..., None, :] + self.corners
        valid = ((indices >= 0) & (indices < size_xyz.long())).all(dim=-1)
        weights = torch.where(self.corners.bool(), fraction[..., None, :], 1 - fraction[..., None, :])
        weights = weights.prod(dim=-1) * valid * (length[:, None, None] / self.n_samples)
        indices = torch.maximum(indices, torch.zeros_like(indices))
        indices = torch.minimum(indices, size_xyz.long() - 1)
        flat_indices = indices[..., 0] + self.volume_shape[2] * (
            indices[..., 1] + self.volume_shape[1] * indices[..., 2]
        )
        return flat_indices, weights

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Integrate trilinearly interpolated attenuation along each source-detector segment."""
        return self._project(volume, self.source_positions, self.detector_positions)

    def _project(self, volume: torch.Tensor, sources: torch.Tensor, detectors: torch.Tensor) -> torch.Tensor:
        self._check_input(volume, self.volume_shape)
        batch = volume.shape[0]
        flat_volume = volume.reshape(batch, -1)
        ray_count = detectors.numel() // 3
        chunks = []
        for start in range(0, ray_count, self.ray_chunk):
            indices, weights = self._stencil(start, min(start + self.ray_chunk, ray_count), sources, detectors)
            values = flat_volume[:, indices.flatten()].reshape(batch, *indices.shape)
            chunks.append((values * weights).sum(dim=(-1, -2)))
        return torch.cat(chunks, dim=-1).reshape(batch, 1, *self.projection_shape)

    def adjoint(self, projections: torch.Tensor) -> torch.Tensor:
        """Apply the exact discrete transpose, with no angular or reconstruction weighting."""
        return self._backproject(projections, self.source_positions, self.detector_positions)

    def _backproject(self, projections: torch.Tensor, sources: torch.Tensor, detectors: torch.Tensor) -> torch.Tensor:
        self._check_input(projections, self.projection_shape)
        batch = projections.shape[0]
        flat_projections = projections.reshape(batch, -1)
        image = projections.new_zeros(batch, self.volume_shape[0] * self.volume_shape[1] * self.volume_shape[2])
        for start in range(0, flat_projections.shape[1], self.ray_chunk):
            end = min(start + self.ray_chunk, flat_projections.shape[1])
            indices, weights = self._stencil(start, end, sources, detectors)
            values = flat_projections[:, start:end, None, None] * weights
            image = image.scatter_add(1, indices.flatten()[None, :].expand(batch, -1), values.reshape(batch, -1))
        return image.reshape(batch, 1, *self.volume_shape)

    def backward(self, projections: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`adjoint`, consistent with the 2D projectors."""
        return self.adjoint(projections)
