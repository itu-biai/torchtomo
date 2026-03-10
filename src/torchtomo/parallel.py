"""Parallel beam CT projector."""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseProjector
from .filters import FilterType, apply_filter


class ParallelBeam(BaseProjector):
    """
    Parallel beam CT projector.

    In parallel beam geometry, all X-rays are parallel for each projection
    angle. This is the simplest geometry and is used in synchrotron CT.

    Example:
        >>> projector = ParallelBeam(img_size=256, n_angles=180, n_det=256)
        >>> sinogram = projector.forward(image)  # [B, 1, 180, 256]
        >>> recon = projector.fbp(sinogram)      # [B, 1, 256, 256]
    """

    def __init__(
        self,
        img_size: int = 256,
        n_angles: int = 180,
        n_det: Optional[int] = None,
        angle_range: tuple[float, float] = (0, np.pi),
        circle: bool = True,
    ):
        """
        Initialize parallel beam projector.

        Args:
            img_size: Image size (assumed square)
            n_angles: Number of projection angles
            n_det: Number of detector elements (default: img_size)
            angle_range: Range of angles in radians (default: 0 to pi)
            circle: If True, mask image to inscribed circle
        """
        n_det = n_det or img_size
        super().__init__(img_size, n_angles, n_det, angle_range)

        self.circle = circle

        # Pixel size (assuming image spans [-1, 1])
        self.pixel_size = 2.0 / img_size

        coords = torch.linspace(-1, 1, img_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

        # Precompute rotation grids for forward projection
        self.register_buffer("forward_grids", self._precompute_forward_grids(grid_x, grid_y))
        self.register_buffer("backward_grids", self._precompute_backward_grids(grid_x, grid_y))

        # Circle mask for reconstruction
        if circle:
            mask = (grid_x**2 + grid_y**2 <= 1).float()
            self.register_buffer("circle_mask", mask)

    def _precompute_forward_grids(self, grid_x: torch.Tensor, grid_y: torch.Tensor) -> torch.Tensor:
        """Precompute sampling grids for forward projection."""
        grids = []

        for angle in self.angles:
            grid = self._rotation_grid(angle, grid_x, grid_y)
            grids.append(grid)

        return torch.stack(grids)  # [n_angles, H, W, 2]

    def _rotation_grid(self, angle: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor) -> torch.Tensor:
        """Create sampling grid for rotating image by angle."""
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # Rotation matrix (rotate coordinates, not image)
        x_rot = cos_a * grid_x + sin_a * grid_y
        y_rot = -sin_a * grid_x + cos_a * grid_y

        # Stack to grid format [H, W, 2]
        grid = torch.stack([x_rot, y_rot], dim=-1)

        return grid

    def _precompute_backward_grids(self, grid_x: torch.Tensor, grid_y: torch.Tensor) -> torch.Tensor:
        """Precompute detector lookup grids for backprojection."""
        grids = []

        for angle in self.angles:
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            t = grid_x * cos_a - grid_y * sin_a

            grid = torch.zeros(self.img_size, self.img_size, 2)
            grid[..., 0] = t
            grids.append(grid)

        return torch.stack(grids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward projection (Radon transform): image -> sinogram.

        Args:
            x: Image tensor [B, 1, H, W]

        Returns:
            Sinogram [B, 1, n_angles, n_det]
        """
        B = x.shape[0]

        # Apply circle mask if enabled
        if self.circle:
            x = x * self.circle_mask.view(1, 1, self.img_size, self.img_size)

        projections = []
        chunk_size = self._angle_chunk_size(B, self.img_size * self.img_size, x.device)

        for start in range(0, self.n_angles, chunk_size):
            end = min(start + chunk_size, self.n_angles)
            angle_count = end - start
            grid = self.forward_grids[start:end]
            grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
            grid = grid.reshape(B * angle_count, self.img_size, self.img_size, 2)
            batch = x.unsqueeze(1).expand(-1, angle_count, -1, -1, -1)
            batch = batch.reshape(B * angle_count, 1, self.img_size, self.img_size)

            rotated = F.grid_sample(
                batch,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

            projection = rotated.sum(dim=2) * self.pixel_size
            projection = projection.reshape(B, angle_count, 1, self.img_size)
            projections.append(projection.permute(0, 2, 1, 3))

        # Stack to sinogram [B, 1, n_angles, n_det]
        sinogram = torch.cat(projections, dim=2)

        return sinogram

    def backward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Back projection (adjoint of Radon transform): sinogram -> image.

        This smears each projection back across the image.

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

            sino_rows = sinogram[:, :, start:end, :]
            sino_rows = sino_rows.permute(0, 2, 1, 3).reshape(B * angle_count, 1, 1, self.n_det)
            grid = self.backward_grids[start:end]
            grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
            grid = grid.reshape(B * angle_count, self.img_size, self.img_size, 2)

            contribution = F.grid_sample(
                sino_rows,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            contribution = contribution.reshape(B, angle_count, 1, self.img_size, self.img_size)
            recon += contribution.sum(dim=1)

        # Normalize by angular spacing (delta_theta)
        recon = recon * self.angle_step

        # Apply circle mask
        if self.circle:
            recon = recon * self.circle_mask.view(1, 1, self.img_size, self.img_size)

        return recon

    def fbp(self, sinogram: torch.Tensor, filter_name: FilterType = "ramp") -> torch.Tensor:
        """
        Filtered back-projection reconstruction.

        Args:
            sinogram: Sinogram [B, 1, n_angles, n_det]
            filter_name: Filter type ('ramp', 'shepp-logan', 'cosine',
                'hamming', 'hann')

        Returns:
            Reconstructed image [B, 1, H, W]
        """
        # Apply ramp filter in frequency domain
        filtered_sino = apply_filter(sinogram, filter_name)

        # Scale the filtered sinogram by img_size / 2
        # This compensates for the pixel_size scaling in forward projection
        # and the discrete approximation of the FBP integral
        filtered_sino = filtered_sino * (self.img_size / 2)

        # Back-project
        recon = self.backward(filtered_sino)

        return recon
