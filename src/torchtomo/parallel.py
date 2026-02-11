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

        # Precompute rotation grids for forward projection
        grids = self._precompute_forward_grids()
        self.register_buffer("forward_grids", grids)

        # Circle mask for reconstruction
        if circle:
            coords = torch.linspace(-1, 1, img_size)
            y, x = torch.meshgrid(coords, coords, indexing="ij")
            mask = (x**2 + y**2 <= 1).float()
            self.register_buffer("circle_mask", mask)

    def _precompute_forward_grids(self) -> torch.Tensor:
        """Precompute sampling grids for forward projection."""
        grids = []

        for angle in self.angles:
            grid = self._rotation_grid(angle)
            grids.append(grid)

        return torch.stack(grids)  # [n_angles, H, W, 2]

    def _rotation_grid(self, angle: torch.Tensor) -> torch.Tensor:
        """Create sampling grid for rotating image by angle."""
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # Create normalized coordinate grid [-1, 1]
        coords = torch.linspace(-1, 1, self.img_size)
        y, x = torch.meshgrid(coords, coords, indexing="ij")

        # Rotation matrix (rotate coordinates, not image)
        x_rot = cos_a * x + sin_a * y
        y_rot = -sin_a * x + cos_a * y

        # Stack to grid format [H, W, 2]
        grid = torch.stack([x_rot, y_rot], dim=-1)

        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward projection (Radon transform): image -> sinogram.

        Args:
            x: Image tensor [B, 1, H, W]

        Returns:
            Sinogram [B, 1, n_angles, n_det]
        """
        B = x.shape[0]
        device = x.device

        # Apply circle mask if enabled
        if self.circle:
            x = x * self.circle_mask.view(1, 1, self.img_size, self.img_size)

        projections = []

        for i in range(self.n_angles):
            # Get rotation grid for this angle
            grid = self.forward_grids[i].unsqueeze(0).expand(B, -1, -1, -1)
            grid = grid.to(device)

            # Rotate image
            rotated = F.grid_sample(
                x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )

            # Sum along vertical axis (parallel rays) - this is the line integral
            # Multiply by pixel size to get proper integral
            projection = rotated.sum(dim=2) * self.pixel_size  # [B, 1, W]

            projections.append(projection)

        # Stack to sinogram [B, 1, n_angles, n_det]
        sinogram = torch.stack(projections, dim=2)

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
        device = sinogram.device

        recon = torch.zeros(B, 1, self.img_size, self.img_size, device=device)

        for i in range(self.n_angles):
            angle = self.angles[i]
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            # Image coordinates
            coords = torch.linspace(-1, 1, self.img_size, device=device)
            y, x = torch.meshgrid(coords, coords, indexing="ij")

            # Project each pixel onto detector line
            # t = x * cos(angle) + y * sin(angle)
            # Note: flip y to match image coordinate convention
            t = x * cos_a - y * sin_a

            # Create sampling grid for this projection
            # We need to sample from sinogram row at position t
            grid = torch.zeros(B, self.img_size, self.img_size, 2, device=device)
            grid[..., 0] = t  # detector position
            grid[..., 1] = 0  # single row

            # Get sinogram row [B, 1, 1, n_det]
            sino_row = sinogram[:, :, i : i + 1, :]

            # Sample and add to reconstruction
            contribution = F.grid_sample(
                sino_row,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

            recon += contribution

        # Normalize by angular spacing (delta_theta)
        delta_theta = (self.angle_range[1] - self.angle_range[0]) / self.n_angles
        recon = recon * delta_theta

        # Apply circle mask
        if self.circle:
            recon = recon * self.circle_mask.view(1, 1, self.img_size, self.img_size)

        return recon

    def fbp(
        self, sinogram: torch.Tensor, filter_name: FilterType = "ramp"
    ) -> torch.Tensor:
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
