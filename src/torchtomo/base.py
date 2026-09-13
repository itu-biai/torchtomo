"""Base class for CT projectors."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class _DiscreteAdjoint(torch.autograd.Function):
    """Transpose a fixed linear projector without differentiating a VJP twice."""

    @staticmethod
    def forward(ctx, sinogram: torch.Tensor, projector: "BaseProjector") -> torch.Tensor:
        ctx.projector = projector
        # The VJP is needed even during evaluation under no_grad/inference_mode.
        # Its temporary graph is freed here; training uses the explicit rule below.
        with torch.inference_mode(False), torch.enable_grad():
            image = torch.zeros(
                sinogram.shape[0],
                1,
                projector.img_size,
                projector.img_size,
                device=sinogram.device,
                dtype=sinogram.dtype,
                requires_grad=True,
            )
            projection = projector.forward(image)
            return torch.autograd.grad(projection, image, sinogram, create_graph=False)[0]

    @staticmethod
    def backward(ctx, grad_image: torch.Tensor):
        # d(A^T y)/dy = A^T, so its reverse-mode product is A grad_image.
        # This avoids grid_sample double backward on PyTorch versions lacking it.
        return ctx.projector.forward(grad_image), None


class BaseProjector(nn.Module, ABC):
    """
    Abstract base class for CT projectors.

    All projectors support:
        - forward(): Image -> Sinogram (Radon transform)
        - backward()/adjoint(): Sinogram -> Image (Exact discrete adjoint)
        - backproject(): Sinogram -> Image (Analytical backprojection for FBP)
        - fbp(): Filtered back-projection reconstruction

    All operations are differentiable with respect to their tensor inputs.
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

    def adjoint(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Exact discrete adjoint of forward() for Euclidean tensor inner products.

        For fixed, real, linear geometry, computes A^T y from the forward VJP,
        including its interpolation, integration weights, and circle mask.
        No angular normalization or FBP weighting is added. Equality of inner
        products holds up to floating-point roundoff.

        Differentiable with respect to sinogram, with backward gradient A g,
        so this operator can be used inside Learned Primal-Dual networks.
        Geometry must remain fixed between evaluation and backpropagation;
        gradients with respect to geometry are not supported. Projector buffers
        must have the same device and dtype as sinogram, as for forward().

        This reference implementation builds and differentiates a temporary
        forward graph on each call, which can cost more memory and time than
        backproject(). It also works under no_grad() and inference_mode().

        Args:
            sinogram: Sinogram of shape [B, 1, n_angles, n_det]

        Returns:
            Adjoint image of shape [B, 1, img_size, img_size]
        """
        return _DiscreteAdjoint.apply(sinogram, self)

    def backward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Exact discrete adjoint of forward(); alias for adjoint().

        Differentiable with respect to sinogram: the reverse-mode gradient is
        forward(grad_image). For the historical FBP backprojection, which has
        different interpolation and normalization, use backproject().
        """
        return self.adjoint(sinogram)

    @abstractmethod
    def backproject(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Analytical backprojection used by fbp(): sinogram -> image.

        Includes reconstruction normalization and is not the exact discrete
        adjoint of forward(). Use adjoint() for iterative/learned algorithms
        requiring a matched operator pair, or fbp() for reconstruction.

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
