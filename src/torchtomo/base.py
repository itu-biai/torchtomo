"""Base class for CT projectors."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ._cuda_kernels import cuda_kernels_available
from ._nvrtc import runtime_available
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


class _KernelProject(torch.autograd.Function):
    """A x on the runtime-compiled CUDA kernels; the VJP is their matched adjoint."""

    @staticmethod
    def forward(ctx, image: torch.Tensor, projector: "BaseProjector") -> torch.Tensor:
        ctx.projector = projector
        return projector._kernel_forward(image)

    @staticmethod
    def backward(ctx, grad_sinogram: torch.Tensor):
        # Through apply, so the gradient itself is differentiable (double backward).
        return _KernelAdjoint.apply(grad_sinogram, ctx.projector), None


class _KernelAdjoint(torch.autograd.Function):
    """A^T y on the runtime-compiled CUDA kernels; the VJP is the kernel forward."""

    @staticmethod
    def forward(ctx, sinogram: torch.Tensor, projector: "BaseProjector") -> torch.Tensor:
        ctx.projector = projector
        return projector._kernel_adjoint(sinogram)

    @staticmethod
    def backward(ctx, grad_image: torch.Tensor):
        return _KernelProject.apply(grad_image, ctx.projector), None


class _KernelBackproject(torch.autograd.Function):
    """FBP backprojection on the CUDA kernels; the VJP differentiates the PyTorch path."""

    @staticmethod
    def forward(ctx, sinogram: torch.Tensor, projector: "BaseProjector") -> torch.Tensor:
        ctx.projector = projector
        ctx.sino_shape = tuple(sinogram.shape)
        return projector._kernel_backproject(sinogram)

    @staticmethod
    def backward(ctx, grad_image: torch.Tensor):
        with torch.inference_mode(False), torch.enable_grad():
            sino = torch.zeros(ctx.sino_shape, device=grad_image.device, dtype=grad_image.dtype, requires_grad=True)
            recon = ctx.projector._backproject_eager(sino)
            grad_sino = torch.autograd.grad(recon, sino, grad_image.contiguous())[0]
        return grad_sino, None


def _check_backend(backend: str, choices: tuple[str, ...]) -> str:
    """Validate a backend name and settle "auto" on what this machine has.

    "auto" answers one question, whether NVRTC and the CUDA driver load here, and
    it answers it once: a projector reports the backend it will actually use, and
    a CPU or float64 tensor still falls back from "cuda" call by call.
    """
    if backend not in choices:
        raise ValueError(f"backend must be one of {choices}, got {backend!r}")
    if backend == "auto":
        return "cuda" if runtime_available() else "torch"
    return backend


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
        self.backend = "torch"
        # Small float32 tables for the CUDA kernels, keyed by device; see _kernel_cached.
        self._kernel_cache: dict = {}
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
        self._kernel_cache = {}
        return result

    def _use_kernels(self, tensor: torch.Tensor) -> bool:
        """backend='cuda' on a float32 CUDA tensor with NVRTC available; else the PyTorch path."""
        return self.backend == "cuda" and cuda_kernels_available(tensor.device, tensor.dtype)

    def _kernel_cached(self, name: str, device: torch.device, build):
        """Build a kernel table once per device, as ordinary tensors even under inference_mode."""
        key = (name, device)
        value = self._kernel_cache.get(key)
        if value is None:
            with torch.inference_mode(False), torch.no_grad():
                value = build(device)
            self._kernel_cache[key] = value
        return value

    def _kernel_trig(self, device: torch.device) -> torch.Tensor:
        """[n_angles, 2] float32 (cos, sin), computed in float64 from the angle buffer."""

        def build(device):
            angles = self.angles.detach().to(device=device, dtype=torch.float64)
            return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).to(torch.float32).contiguous()

        return self._kernel_cached("trig", device, build)

    def _kernel_coords(self, device: torch.device) -> torch.Tensor:
        """Pixel centres in normalised coordinates, as the eager grids use them."""
        return self._kernel_cached(
            "coords", device, lambda device: torch.linspace(-1, 1, self.img_size, dtype=torch.float32, device=device)
        )

    def _kernel_mask(self, device: torch.device) -> torch.Tensor | None:
        if not getattr(self, "circle", False):
            return None
        return self._kernel_cached(
            "mask", device, lambda device: self.circle_mask.detach().to(device=device, dtype=torch.float32).contiguous()
        )

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

        Parallel beam and fan beam on CPU and CUDA call grid_sample's input
        backward kernel directly. MPS and older PyTorch builds fall back to a
        throwaway forward VJP. Both paths work under no_grad() and inference_mode().
        With backend='cuda' this is the gather kernel that transposes the CUDA
        forward, which is then the operator it is the exact adjoint of.

        Args:
            sinogram: Sinogram of shape [B, 1, n_angles, n_det]

        Returns:
            Adjoint image of shape [B, 1, img_size, img_size]
        """
        if self._use_kernels(sinogram):
            return _KernelAdjoint.apply(sinogram, self)
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
