"""Base class for CT projectors."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ._sampling import grid_sample_input_backward_supported


def _vjp_adjoint(projector: "BaseProjector", sinogram: torch.Tensor) -> torch.Tensor:
    """A^T y by differentiating a throwaway forward. Fallback and test reference."""
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


def _adjoint_image(projector: "BaseProjector", sinogram: torch.Tensor) -> torch.Tensor:
    """A^T y: the projector's direct kernel when it has one, otherwise the VJP."""
    direct = getattr(projector, "_direct_adjoint", None)
    if direct is not None and grid_sample_input_backward_supported(sinogram.device):
        return direct(sinogram)
    return _vjp_adjoint(projector, sinogram)


class _DiscreteAdjoint(torch.autograd.Function):
    """Transpose a fixed linear projector without differentiating a VJP twice."""

    @staticmethod
    def forward(ctx, sinogram: torch.Tensor, projector: "BaseProjector") -> torch.Tensor:
        ctx.projector = projector
        # Evaluation under no_grad/inference_mode still needs A^T. The direct kernel
        # never builds a graph; the VJP fallback re-enables grad inside itself.
        return _adjoint_image(projector, sinogram)

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
        angles: torch.Tensor | None = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.n_det = n_det
        self.angle_range = angle_range
        if angles is not None:
            source = torch.as_tensor(angles, dtype=torch.float64).reshape(-1).detach().cpu().contiguous()
            if source.numel() < 1:
                raise ValueError("angles must contain at least one value")
            self.n_angles = int(source.numel())
            self._explicit_angles = source
            span = float(source[-1] - source[0]) if self.n_angles > 1 else 0.0
            self.angle_step = span / max(self.n_angles - 1, 1)
        else:
            self.n_angles = n_angles
            self._explicit_angles = None
            self.angle_step = (angle_range[1] - angle_range[0]) / n_angles
        self._set_angle_buffer(torch.device("cpu"), torch.float32)

    def _set_angle_buffer(self, device: torch.device, dtype: torch.dtype) -> None:
        """Rebuild angles in the requested dtype from the original range or list.

        The default list is an open interval: n samples, spacing
        (end - start) / n, so both endpoints of a half-turn are not included.
        An explicit list is stored in float64 and cast on `.to()`.
        """
        if self._explicit_angles is not None:
            angles = self._explicit_angles.to(device=device, dtype=dtype)
        else:
            start, end = float(self.angle_range[0]), float(self.angle_range[1])
            step = (end - start) / self.n_angles
            angles = torch.arange(self.n_angles, dtype=dtype, device=device) * step + start
        self.register_buffer("angles", angles)

    def _apply(self, fn, *args, **kwargs):
        result = super()._apply(fn, *args, **kwargs)
        self._set_angle_buffer(self.angles.device, self.angles.dtype)
        return result

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

        For fixed, real, linear geometry, computes A^T y including interpolation,
        integration weights, and circle mask. No angular normalization or FBP
        weighting is added. Equality of inner products holds up to floating-point
        roundoff.

        Differentiable with respect to sinogram, with backward gradient A g,
        so this operator can be used inside Learned Primal-Dual networks.
        Geometry must remain fixed between evaluation and backpropagation;
        gradients with respect to geometry are not supported. Projector buffers
        must have the same device and dtype as sinogram, as for forward().

        Parallel beam on CPU and CUDA calls grid_sample's input backward kernel
        directly. Other geometries, MPS, and older PyTorch builds fall back to a
        throwaway forward VJP. Both paths work under no_grad() and inference_mode().

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
