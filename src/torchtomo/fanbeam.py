"""Fan beam CT projector with flat detector."""

from typing import Optional

import numpy as np
import torch

from ._sampling import grid_sample_input_backward, sample_bilinear
from .base import BaseProjector
from .filters import FilterType, apply_filter


class FanBeam(BaseProjector):
    """
    Fan beam CT projector with flat detector.

    In fan beam geometry, X-rays emanate from a point source and spread
    out in a fan shape to a flat detector array. This is the standard geometry
    for clinical CT scanners.

    Geometry:
        - Point source at distance src_dist from origin
        - Flat detector at distance det_dist from origin (opposite side)
        - Source and detector rotate around the object

    Distances default to twice the image size and the detector to 1.5 samples
    per pixel at the isocentre, so a 512 px projector has src = det = 1024 px
    and 768 bins. Rays are clipped to the unit circle even when circle=False.

    Example:
        >>> projector = FanBeam(img_size=256, n_angles=360)
        >>> sinogram = projector.forward(image)
        >>> recon = projector.fbp(sinogram)
    """

    def __init__(
        self,
        img_size: int = 256,
        n_angles: int = 360,
        n_det: Optional[int] = None,
        src_dist: Optional[float] = None,
        det_dist: Optional[float] = None,
        det_width: Optional[float] = None,
        det_spacing: Optional[float] = None,
        angle_range: tuple[float, float] = (0, 2 * np.pi),
        n_samples: Optional[int] = None,
        circle: bool = True,
        angles: Optional[torch.Tensor] = None,
    ):
        """
        Initialize fan beam projector.

        Args:
            img_size: Image size (assumed square)
            n_angles: Number of projection angles
            n_det: Detector bins (default: round(1.5 * img_size))
            src_dist: Source to isocenter distance in pixels (default: 2 * img_size)
            det_dist: Isocenter to detector distance in pixels (default: 2 * img_size)
            det_width: Total detector width (alternative to det_spacing)
            det_spacing: Spacing between detector elements (alternative to det_width)
            angle_range: Range of angles (default: full rotation)
            n_samples: Samples per ray (default: img_size)
            circle: If True, mask image to inscribed circle. Rays are still
                clipped to the unit circle when this is False.
            angles: Explicit angle samples in radians. When omitted, n_angles
                samples cover [start, end) with spacing (end - start) / n_angles.
        """
        src_dist = float(2 * img_size if src_dist is None else src_dist)
        det_dist = float(2 * img_size if det_dist is None else det_dist)
        n_det = int(round(1.5 * img_size) if n_det is None else n_det)
        n_samples = int(img_size if n_samples is None else n_samples)
        super().__init__(img_size, n_angles, n_det, angle_range, angles=angles)

        self.src_dist = src_dist
        self.det_dist = det_dist
        magnification = (src_dist + det_dist) / src_dist
        if det_spacing is not None:
            self.det_width = det_spacing * n_det
        elif det_width is not None:
            self.det_width = det_width
        else:
            self.det_width = 1.5 * magnification * img_size
        self.n_samples = n_samples
        self.circle = circle

        self.scale = 2.0 / img_size
        self._src_dist_norm = src_dist * self.scale
        self._det_dist_norm = det_dist * self.scale
        self._det_width_norm = self.det_width * self.scale
        self._set_geometry_buffers()

    def _apply(self, fn, *args, **kwargs):
        result = super()._apply(fn, *args, **kwargs)
        self._set_geometry_buffers()
        return result

    def _set_geometry_buffers(self) -> None:
        """Rebuild ray and backprojection grids in the current angles dtype."""
        dtype, device = self.angles.dtype, self.angles.device
        ray_grids, ray_lengths = self._precompute_ray_grids()
        self.register_buffer("ray_grids", ray_grids)
        self.register_buffer("ray_lengths", ray_lengths)

        back_grids, weights = self._precompute_backward_grids()
        self.register_buffer("backward_grids", back_grids)
        self.register_buffer("backward_weights", weights)

        det_pos = torch.linspace(
            -self._det_width_norm / 2, self._det_width_norm / 2, self.n_det, dtype=dtype, device=device
        )
        D = self._src_dist_norm + self._det_dist_norm
        cos_weight = D / torch.sqrt(D**2 + det_pos**2)
        self.register_buffer("cos_weight", cos_weight)

        if self.circle:
            coords = torch.linspace(-1, 1, self.img_size, dtype=dtype, device=device)
            y, x = torch.meshgrid(coords, coords, indexing="ij")
            mask = (x**2 + y**2 <= 1).to(dtype=dtype)
            self.register_buffer("circle_mask", mask)

    def _precompute_ray_grids(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sampling grids [n_angles, n_det, n_samples, 2] and path lengths [n_angles, n_det]."""
        dtype, device = self.angles.dtype, self.angles.device
        cos_a = torch.cos(self.angles).view(-1, 1)
        sin_a = torch.sin(self.angles).view(-1, 1)
        src_x = -self._src_dist_norm * sin_a
        src_y = self._src_dist_norm * cos_a
        det_cx = self._det_dist_norm * sin_a
        det_cy = -self._det_dist_norm * cos_a
        det_offsets = torch.linspace(
            -self._det_width_norm / 2, self._det_width_norm / 2, self.n_det, dtype=dtype, device=device
        ).view(1, -1)
        det_x = det_cx + det_offsets * cos_a
        det_y = det_cy + det_offsets * sin_a
        dir_x = det_x - src_x
        dir_y = det_y - src_y
        ray_len_full = torch.sqrt(dir_x**2 + dir_y**2)
        dir_x = dir_x / ray_len_full
        dir_y = dir_y / ray_len_full
        a = dir_x**2 + dir_y**2
        b = 2 * (src_x * dir_x + src_y * dir_y)
        c = src_x**2 + src_y**2 - 1.0
        disc = torch.clamp(b**2 - 4 * a * c, min=0)
        sqrt_disc = torch.sqrt(disc)
        t_entry = torch.clamp((-b - sqrt_disc) / (2 * a), min=0)
        t_exit = torch.clamp((-b + sqrt_disc) / (2 * a), min=t_entry)
        ray_lengths = t_exit - t_entry
        t_samples = torch.linspace(0, 1, self.n_samples, dtype=dtype, device=device).view(1, 1, -1)
        t_actual = t_entry.unsqueeze(-1) + t_samples * ray_lengths.unsqueeze(-1)
        ray_x = src_x.unsqueeze(-1) + t_actual * dir_x.unsqueeze(-1)
        ray_y = src_y.unsqueeze(-1) + t_actual * dir_y.unsqueeze(-1)
        return torch.stack([ray_x, ray_y], dim=-1), ray_lengths

    def _precompute_backward_grids(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Backprojection grids [n_angles, H, W, 2] and 1/U² weights [n_angles, H, W]."""
        dtype, device = self.angles.dtype, self.angles.device
        coords = torch.linspace(-1, 1, self.img_size, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        cos_a = torch.cos(self.angles).view(-1, 1, 1)
        sin_a = torch.sin(self.angles).view(-1, 1, 1)
        src_x = -self._src_dist_norm * sin_a
        src_y = self._src_dist_norm * cos_a
        det_cx = self._det_dist_norm * sin_a
        det_cy = -self._det_dist_norm * cos_a
        px_x = grid_x - src_x
        px_y = grid_y - src_y
        sd_x = det_cx - src_x
        sd_y = det_cy - src_y
        sd_len = torch.sqrt(sd_x**2 + sd_y**2)
        sd_ux = sd_x / sd_len
        sd_uy = sd_y / sd_len
        proj_len = px_x * sd_ux + px_y * sd_uy
        t = sd_len / (proj_len + 1e-8)
        int_x = src_x + t * px_x
        int_y = src_y + t * px_y
        det_offset = (int_x - det_cx) * cos_a + (int_y - det_cy) * sin_a
        det_normalized = det_offset / (self._det_width_norm / 2)
        grid = torch.stack([det_normalized, torch.zeros_like(det_normalized)], dim=-1)
        src = self._src_dist_norm
        U = (src + grid_x * sin_a - grid_y * cos_a) / src
        weight = 1.0 / U.clamp_min(1e-6).square()
        return grid, weight

    def _backward_grid(self, start: int, end: int) -> torch.Tensor:
        """Detector sampling grids for one chunk of angles, shape [count, H, W, 2]."""
        return self.backward_grids[start:end]

    def _direct_adjoint(self, sinogram: torch.Tensor) -> torch.Tensor:
        """A^T y by calling grid_sample's input backward, skipping a throwaway forward."""
        batch = sinogram.shape[0]
        size = self.img_size
        chunk = self._angle_chunk_size(batch, self.n_det * self.n_samples, sinogram.device)
        out = torch.zeros(1, batch, size, size, device=sinogram.device, dtype=sinogram.dtype)
        shape_only = torch.empty(1, batch, size, size, device=sinogram.device, dtype=sinogram.dtype)
        scale = self.ray_lengths / self.n_samples
        for start in range(0, self.n_angles, chunk):
            end = min(start + chunk, self.n_angles)
            angle_count = end - start
            grid = self.ray_grids[start:end].reshape(1, angle_count * self.n_det, self.n_samples, 2)
            grad = (sinogram[:, 0, start:end, :] * scale[start:end]).reshape(1, batch, angle_count * self.n_det, 1)
            grad = grad.expand(1, batch, angle_count * self.n_det, self.n_samples)
            out += grid_sample_input_backward(grad, shape_only, grid)
        out = out.view(batch, 1, size, size)
        if self.circle:
            out = out * self.circle_mask.view(1, 1, size, size)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward projection: image -> sinogram.

        Args:
            x: Image tensor [B, 1, H, W]

        Returns:
            Sinogram [B, 1, n_angles, n_det]
        """
        B = x.shape[0]

        if self.circle:
            x = x * self.circle_mask.view(1, 1, self.img_size, self.img_size)

        size = self.img_size
        projections = []
        chunk_size = self._angle_chunk_size(B, self.n_det * self.n_samples, x.device)

        for start in range(0, self.n_angles, chunk_size):
            end = min(start + chunk_size, self.n_angles)
            angle_count = end - start
            grid = self.ray_grids[start:end].reshape(1, angle_count * self.n_det, self.n_samples, 2)
            samples = sample_bilinear(x.reshape(1, B, size, size), grid)
            projection = samples.view(B, angle_count, self.n_det, self.n_samples).mean(dim=-1)
            projection = projection * self.ray_lengths[start:end]
            projections.append(projection.unsqueeze(1))

        return torch.cat(projections, dim=2)

    def backproject(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Weighted analytical backprojection for FBP: sinogram -> image.

        Includes distance weights and angular normalization for reconstruction.
        This is not the exact discrete adjoint of forward(); use backward()
        or adjoint() when a matched discrete operator pair is required.

        Args:
            sinogram: Sinogram [B, 1, n_angles, n_det]

        Returns:
            Back-projected image [B, 1, H, W]
        """
        B = sinogram.shape[0]
        recon = torch.zeros(B, 1, self.img_size, self.img_size, device=sinogram.device)
        chunk_size = self._angle_chunk_size(B, self.img_size * self.img_size, sinogram.device)

        for start in range(0, self.n_angles, chunk_size):
            end = min(start + chunk_size, self.n_angles)
            angle_count = end - start
            rows = sinogram[:, 0, start:end, :].permute(1, 0, 2).reshape(angle_count, B, 1, self.n_det)
            contrib = sample_bilinear(rows, self.backward_grids[start:end])
            weight = self.backward_weights[start:end].unsqueeze(1)
            recon += (contrib * weight).sum(dim=0).unsqueeze(1)

        recon = recon * self.angle_step / 2

        # Circle mask
        if self.circle:
            recon = recon * self.circle_mask.view(1, 1, self.img_size, self.img_size)

        return recon

    def fbp(self, sinogram: torch.Tensor, filter_name: FilterType = "ramp") -> torch.Tensor:
        """
        Filtered back-projection for fan beam with flat detector.

        Args:
            sinogram: Sinogram [B, 1, n_angles, n_det]
            filter_name: Filter type

        Returns:
            Reconstructed image [B, 1, H, W]
        """
        cos_w = self.cos_weight.view(1, 1, 1, -1)
        weighted_sino = sinogram * cos_w

        filtered_sino = apply_filter(weighted_sino, filter_name)

        # Ramp is built in bin units; convert to the virtual detector at the isocentre.
        mag = (self.src_dist + self.det_dist) / self.src_dist
        virt_px = self.det_width / self.n_det / mag
        filtered_sino = filtered_sino * (self.img_size / 2) / virt_px

        recon = self.backproject(filtered_sino)

        return recon

    def __repr__(self) -> str:
        return (
            f"FanBeam("
            f"img_size={self.img_size}, "
            f"n_angles={self.n_angles}, "
            f"n_det={self.n_det}, "
            f"src_dist={self.src_dist}, "
            f"det_dist={self.det_dist})"
        )
