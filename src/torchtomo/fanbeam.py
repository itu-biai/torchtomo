"""Fan beam CT projector with flat detector."""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

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

    Example:
        >>> projector = FanBeam(
        ...     img_size=256,
        ...     n_angles=360,
        ...     n_det=400,
        ...     src_dist=500,
        ...     det_dist=500
        ... )
        >>> sinogram = projector.forward(image)
        >>> recon = projector.fbp(sinogram)
    """

    def __init__(
        self,
        img_size: int = 256,
        n_angles: int = 360,
        n_det: int = 400,
        src_dist: float = 500.0,
        det_dist: float = 500.0,
        det_width: Optional[float] = None,
        det_spacing: Optional[float] = None,
        angle_range: tuple[float, float] = (0, 2 * np.pi),
        n_samples: int = 512,
        circle: bool = True,
    ):
        """
        Initialize fan beam projector.

        Args:
            img_size: Image size (assumed square)
            n_angles: Number of projection angles
            n_det: Number of detector elements
            src_dist: Source to isocenter distance (in pixels)
            det_dist: Isocenter to detector distance (in pixels)
            det_width: Total detector width (alternative to det_spacing)
            det_spacing: Spacing between detector elements (alternative to det_width)
            angle_range: Range of angles (default: full rotation)
            n_samples: Number of samples per ray for integration
            circle: If True, mask image to inscribed circle
        """
        super().__init__(img_size, n_angles, n_det, angle_range)

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

        ray_grids, ray_lengths = self._precompute_ray_grids()
        self.register_buffer("ray_grids", ray_grids)
        self.register_buffer("ray_lengths", ray_lengths)

        back_grids, weights = self._precompute_backward_grids()
        self.register_buffer("backward_grids", back_grids)
        self.register_buffer("backward_weights", weights)

        det_pos = torch.linspace(
            -self._det_width_norm / 2, self._det_width_norm / 2, n_det
        )
        D = self._src_dist_norm + self._det_dist_norm
        cos_weight = D / torch.sqrt(D**2 + det_pos**2)
        self.register_buffer("cos_weight", cos_weight)

        if circle:
            coords = torch.linspace(-1, 1, img_size)
            y, x = torch.meshgrid(coords, coords, indexing="ij")
            mask = (x**2 + y**2 <= 1).float()
            self.register_buffer("circle_mask", mask)

    def _precompute_ray_grids(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute sampling grids for all rays.

        Returns:
            grids: [n_angles, n_det, n_samples, 2]
            ray_lengths: [n_angles, n_det] path length through image for each ray
        """
        all_grids = []
        all_lengths = []

        for angle in self.angles:
            grid, lengths = self._compute_rays_for_angle(angle)
            all_grids.append(grid)
            all_lengths.append(lengths)

        return torch.stack(all_grids), torch.stack(all_lengths)

    def _compute_rays_for_angle(
        self, angle: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ray sampling points for a single angle.

        Only samples the portion of each ray that intersects the unit circle
        (image region in normalized coordinates).

        Returns:
            grid: [n_det, n_samples, 2]
            ray_lengths: [n_det] path length through image for each ray
        """
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        src_x = -self._src_dist_norm * sin_a
        src_y = self._src_dist_norm * cos_a

        det_cx = self._det_dist_norm * sin_a
        det_cy = -self._det_dist_norm * cos_a

        det_dir_x = cos_a
        det_dir_y = sin_a

        det_offsets = torch.linspace(
            -self._det_width_norm / 2, self._det_width_norm / 2, self.n_det
        )

        det_x = det_cx + det_offsets * det_dir_x
        det_y = det_cy + det_offsets * det_dir_y

        dir_x = det_x - src_x
        dir_y = det_y - src_y
        ray_len_full = torch.sqrt(dir_x**2 + dir_y**2)
        dir_x = dir_x / ray_len_full
        dir_y = dir_y / ray_len_full

        a = dir_x**2 + dir_y**2
        b = 2 * (src_x * dir_x + src_y * dir_y)
        c = src_x**2 + src_y**2 - 1.0

        discriminant = b**2 - 4 * a * c
        discriminant = torch.clamp(discriminant, min=0)

        sqrt_disc = torch.sqrt(discriminant)
        t_entry = (-b - sqrt_disc) / (2 * a)
        t_exit = (-b + sqrt_disc) / (2 * a)

        t_entry = torch.clamp(t_entry, min=0)
        t_exit = torch.clamp(t_exit, min=t_entry)

        ray_lengths = t_exit - t_entry

        t_samples = torch.linspace(0, 1, self.n_samples).view(1, -1)
        t_actual = t_entry.view(-1, 1) + t_samples * (t_exit - t_entry).view(-1, 1)

        ray_x = src_x + t_actual * dir_x.view(-1, 1)
        ray_y = src_y + t_actual * dir_y.view(-1, 1)

        grid = torch.stack([ray_x, ray_y], dim=-1)

        return grid, ray_lengths

    def _precompute_backward_grids(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute grids and weights for back-projection.

        Returns:
            grids: [n_angles, H, W, 2] sampling positions in sinogram
            weights: [n_angles, H, W] distance weighting
        """
        grids = []
        weights = []

        coords = torch.linspace(-1, 1, self.img_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

        for angle in self.angles:
            grid, weight = self._compute_backward_for_angle(angle, grid_x, grid_y)
            grids.append(grid)
            weights.append(weight)

        return torch.stack(grids), torch.stack(weights)

    def _compute_backward_for_angle(
        self,
        angle: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute back-projection mapping for a single angle.

        For each pixel, determine which detector element it maps to and
        compute the FBP weight U² where U = D / (D + x*sin(β) - y*cos(β)).
        """
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

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

        det_dir_x = cos_a
        det_dir_y = sin_a

        det_offset = (int_x - det_cx) * det_dir_x + (int_y - det_cy) * det_dir_y

        det_normalized = det_offset / (self._det_width_norm / 2)

        grid = torch.zeros(self.img_size, self.img_size, 2)
        grid[..., 0] = det_normalized
        grid[..., 1] = 0

        D = self._src_dist_norm + self._det_dist_norm
        U = D / (D + grid_x * sin_a + grid_y * cos_a)
        weight = U * U

        return grid, weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward projection: image -> sinogram.

        Args:
            x: Image tensor [B, 1, H, W]

        Returns:
            Sinogram [B, 1, n_angles, n_det]
        """
        B = x.shape[0]
        device = x.device

        if self.circle:
            x = x * self.circle_mask.view(1, 1, self.img_size, self.img_size)

        projections = []

        for i in range(self.n_angles):
            grid = self.ray_grids[i]
            ray_len = self.ray_lengths[i]

            grid = grid.unsqueeze(0).expand(B, -1, -1, -1).to(device)

            samples = F.grid_sample(
                x,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

            projection = samples.mean(dim=-1)
            projection = projection * ray_len.view(1, 1, -1).to(device)
            projections.append(projection)

        sinogram = torch.stack(projections, dim=2)

        return sinogram

    def backward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Back projection (adjoint): sinogram -> image.

        Args:
            sinogram: Sinogram [B, 1, n_angles, n_det]

        Returns:
            Back-projected image [B, 1, H, W]
        """
        B = sinogram.shape[0]
        device = sinogram.device

        recon = torch.zeros(B, 1, self.img_size, self.img_size, device=device)

        for i in range(self.n_angles):
            # Sinogram row [B, 1, 1, n_det]
            sino_row = sinogram[:, :, i : i + 1, :]

            # Sampling grid
            grid = self.backward_grids[i].unsqueeze(0).expand(B, -1, -1, -1)
            grid = grid.to(device)

            # Sample
            contribution = F.grid_sample(
                sino_row,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

            # Apply distance weighting
            weight = self.backward_weights[i].view(1, 1, self.img_size, self.img_size)
            weight = weight.to(device)

            recon += contribution * weight

        delta_beta = (self.angle_range[1] - self.angle_range[0]) / self.n_angles
        recon = recon * delta_beta / 2

        # Circle mask
        if self.circle:
            recon = recon * self.circle_mask.view(1, 1, self.img_size, self.img_size)

        return recon

    def fbp(
        self, sinogram: torch.Tensor, filter_name: FilterType = "ramp"
    ) -> torch.Tensor:
        """
        Filtered back-projection for fan beam with flat detector.

        Args:
            sinogram: Sinogram [B, 1, n_angles, n_det]
            filter_name: Filter type

        Returns:
            Reconstructed image [B, 1, H, W]
        """
        cos_w = self.cos_weight.view(1, 1, 1, -1).to(sinogram.device)
        weighted_sino = sinogram * cos_w

        filtered_sino = apply_filter(weighted_sino, filter_name)

        filtered_sino = filtered_sino * (self.img_size / 2)

        recon = self.backward(filtered_sino)

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
