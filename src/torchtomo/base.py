"""Base class for CT projectors."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseProjector(nn.Module, ABC):
    """
    Abstract base class for CT projectors.

    All projectors support:
        - forward(): Image -> Sinogram (Radon transform)
        - backward(): Sinogram -> Image (Adjoint/back-projection)
        - fbp(): Filtered back-projection reconstruction

    All operations are differentiable.
    """

    def __init__(
        self,
        img_size: int,
        n_angles: int,
        n_det: int,
        angle_range: tuple[float, float] = (0, torch.pi),
    ):
        super().__init__()
        self.img_size = img_size
        self.n_angles = n_angles
        self.n_det = n_det
        self.angle_range = angle_range
        self.angle_step = (angle_range[1] - angle_range[0]) / n_angles
        angles = torch.linspace(angle_range[0], angle_range[1], n_angles, dtype=torch.float32)
        self.register_buffer("angles", angles)

    def _angle_chunk_size(self, batch_size: int, work_items: int, device: torch.device) -> int:
        """Bound per-call tensor expansion while still batching angles."""
        target_items = 1 << (24 if device.type != "cpu" else 22)
        per_angle_items = max(1, batch_size * work_items)
        chunk = target_items // per_angle_items
        return max(1, min(self.n_angles, chunk))

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward projection: image -> sinogram.

        Args:
            x: Image tensor of shape [B, 1, H, W]

        Returns:
            Sinogram of shape [B, 1, n_angles, n_det]
        """
        pass

    @abstractmethod
    def backward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Back projection (adjoint): sinogram -> image.

        Note: This is NOT the inverse, just the adjoint operator.
        For reconstruction, use fbp().

        Args:
            sinogram: Sinogram of shape [B, 1, n_angles, n_det]

        Returns:
            Back-projected image of shape [B, 1, H, W]
        """
        pass

    @abstractmethod
    def fbp(self, sinogram: torch.Tensor, filter_name: str = "ramp") -> torch.Tensor:
        """
        Filtered back-projection reconstruction.

        Args:
            sinogram: Sinogram of shape [B, 1, n_angles, n_det]
            filter_name: Filter type ('ramp', 'shepp-logan', 'cosine',
                        'hamming', 'hann')

        Returns:
            Reconstructed image of shape [B, 1, H, W]
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(img_size={self.img_size}, n_angles={self.n_angles}, n_det={self.n_det})"
