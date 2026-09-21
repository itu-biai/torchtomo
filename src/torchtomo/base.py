"""Base class for CT projectors."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ._cuda_kernels import cuda_kernels_available
from ._nvrtc import runtime_available
from ._sampling import grid_sample_input_backward_supported


def _vjp_adjoint(projector: "BaseProjector", sinogram: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """A^T y by differentiating a throwaway forward. Fallback and test reference.

    With create_graph the result stays attached to whatever the forward read, which
    is how the adjoint carries a gradient back to the geometry.
    """
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
        return torch.autograd.grad(projection, image, sinogram, create_graph=create_graph)[0]


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

    # Per-view geometry, one row per view: column 0 is the angle, the rest are
    # lateral offsets in pixels that are zero unless something sets or learns them.
    _POSE_COLUMNS: tuple[str, ...] = ("angle", "detector_shift")

    def __init__(
        self,
        img_size: int,
        n_angles: int,
        n_det: int,
        angle_range: tuple[float, float] = (0, torch.pi),
        angles: torch.Tensor | None = None,
        learnable_geometry: bool = False,
    ):
        super().__init__()
        self.backend = "torch"
        # Small float32 tables for the CUDA kernels, keyed by device; see _kernel_cached.
        self._kernel_cache: dict = {}
        self.img_size = img_size
        self.n_det = n_det
        self.angle_range = angle_range
        self.learnable_geometry = learnable_geometry
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
        self._set_pose(torch.device("cpu"), torch.float32)

    def _angle_samples(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Angles in the requested dtype from the original range or list.

        The default list is an open interval: n samples, spacing
        (end - start) / n, so both endpoints of a half-turn are not included.
        An explicit list is stored in float64 and cast on `.to()`.
        """
        if self._explicit_angles is not None:
            return self._explicit_angles.to(device=device, dtype=dtype)
        start, end = float(self.angle_range[0]), float(self.angle_range[1])
        step = (end - start) / self.n_angles
        return torch.arange(self.n_angles, dtype=dtype, device=device) * step + start

    def _set_pose(self, device: torch.device, dtype: torch.dtype, offsets: torch.Tensor | None = None) -> None:
        """Register the pose table, angles in column 0 and offsets after it."""
        pose = torch.zeros(self.n_angles, len(self._POSE_COLUMNS), device=device, dtype=dtype)
        pose[:, 0] = self._angle_samples(device, dtype)
        if offsets is not None:
            pose[:, 1:] = offsets.to(device=device, dtype=dtype)
        # Whether the offset columns hold anything, so a geometry nobody has shifted
        # keeps the exact arithmetic, and the exact cost, it had before they existed.
        self._pose_has_offsets = bool(offsets is not None and bool(offsets.any()))
        # Angles still as this projector was built; see _apply.
        self._pose_pristine = True
        if self.learnable_geometry:
            self.register_parameter("pose", nn.Parameter(pose))
        else:
            self.register_buffer("pose", pose)

    def set_pose(
        self,
        angles: torch.Tensor | None = None,
        detector_shift: torch.Tensor | float | None = None,
        source_shift: torch.Tensor | float | None = None,
    ) -> None:
        """Write the per-view geometry: angles in radians, shifts in pixels.

        Use this rather than assigning into `pose`, which cannot tell the projector
        that its offsets stopped being zero or that a cached table is now stale.
        """
        columns = {"angle": angles, "detector_shift": detector_shift, "source_shift": source_shift}
        with torch.no_grad():
            for name, value in columns.items():
                if value is None:
                    continue
                if name not in self._POSE_COLUMNS:
                    raise ValueError(f"{type(self).__name__} has no pose column {name!r}")
                column = self._POSE_COLUMNS.index(name)
                self.pose[:, column] = torch.as_tensor(value, device=self.pose.device, dtype=self.pose.dtype)
        self._pose_has_offsets = bool(self.pose[:, 1:].any())
        self._pose_pristine = angles is None and self._pose_pristine
        self._kernel_cache = {}
        self._invalidate_geometry()

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        if name == "pose" and isinstance(value, torch.Tensor):
            # Replacing the table wholesale is the other way to move the geometry,
            # and it is how a pose built as an expression in some other parameter
            # arrives. It has the same consequences as set_pose.
            self._pose_pristine = False
            # A pose that can move is never on the plain path anyway, so do not
            # stop the device to ask whether its offsets are zero at this instant.
            self._pose_has_offsets = True if value.requires_grad else bool(value[:, 1:].any())
            self._kernel_cache = {}
            self._invalidate_geometry()

    def _invalidate_geometry(self) -> None:
        """Drop anything precomputed from the pose; subclasses extend this."""

    @property
    def angles(self) -> torch.Tensor:
        """Projection angles in radians, column 0 of the pose table."""
        return self.pose[:, 0]

    @property
    def detector_shift(self) -> torch.Tensor:
        """Per-view lateral detector offset in pixels, column 1 of the pose table."""
        return self.pose[:, 1]

    @property
    def shift_scale(self) -> float:
        """One pixel of offset in the [-1, 1] coordinates the grids work in.

        The image lattice is linspace(-1, 1, img_size), so a pixel is its spacing,
        which for parallel beam is also exactly one detector bin. Fan beam measures
        its distances against a slightly different pixel and overrides this.
        """
        return 2.0 / max(self.img_size - 1, 1)

    def _geometry_requires_grad(self) -> bool:
        """True when this call has to build the geometry inside the autograd graph."""
        return bool(self.pose.requires_grad) and torch.is_grad_enabled()

    def _geometry_is_mutable(self) -> bool:
        """True when the pose can move, so nothing derived from it may be cached.

        An optimiser step writes into the pose in place and under no_grad, which no
        cache can see. Anything a moving geometry feeds is therefore rebuilt every
        call, including under no_grad, where the gradient itself is not wanted but
        the updated geometry still is.
        """
        return bool(self.pose.requires_grad) or self.learnable_geometry

    def _pose_is_plain(self) -> bool:
        """True for a fixed geometry with no offsets: the arithmetic that predates them.

        `_pose_has_offsets` is a flag, and a flag cannot see an in-place write, so a
        geometry that can move never takes this path even when its offsets read as
        zero. That is what keeps a learnt shift from being silently dropped under
        no_grad.
        """
        return not self._pose_has_offsets and not self._geometry_is_mutable()

    def _apply(self, fn, *args, **kwargs):
        result = super()._apply(fn, *args, **kwargs)
        if self._pose_pristine:
            # Restore the angles from the float64 master so .double() recovers the
            # precision a float32 table lost. Written in place, because a learnable
            # pose is an nn.Parameter an optimiser may already be holding, and only
            # while the angles are still the ones this projector was built with:
            # set_pose() gives them up, and from then on the table is the master.
            with torch.no_grad():
                self.pose[:, 0] = self._angle_samples(self.pose.device, self.pose.dtype)
        self._kernel_cache = {}
        return result

    def _use_kernels(self, tensor: torch.Tensor) -> bool:
        """backend='cuda' on a float32 CUDA tensor with NVRTC available; else the PyTorch path.

        A geometry that wants a gradient takes the PyTorch path, which differentiates
        the sampling grid itself. That covers approximate=True as well, whose inexact
        operator pair would otherwise return a geometry gradient that does not belong
        to the operator it came from. A shifted pose that wants no gradient stays on
        the kernels, which read the shifts from their pose table.
        """
        if self._geometry_requires_grad():
            return False
        return self.backend == "cuda" and cuda_kernels_available(tensor.device, tensor.dtype)

    def _kernel_cached(self, name: str, device: torch.device, build):
        """Build a kernel table once per device, as ordinary tensors even under inference_mode."""
        if self._geometry_is_mutable():
            with torch.inference_mode(False), torch.no_grad():
                return build(device)
        key = (name, device)
        value = self._kernel_cache.get(key)
        if value is None:
            with torch.inference_mode(False), torch.no_grad():
                value = build(device)
            self._kernel_cache[key] = value
        return value

    @property
    def _kernel_shift_unit(self) -> float:
        """One pixel of pose offset in the kernels' length unit: lattice steps here."""
        return 1.0

    def _kernel_pose(self, device: torch.device) -> torch.Tensor:
        """The kernels' view table in float32, built in float64 from the pose.

        [n_angles, 2] of (cos, sin) for a plain pose, which runs the kernels exactly
        as they were before shifts existed. Otherwise [n_angles, 4] with the detector
        and source shift after them, in the kernels' own length unit and zero for a
        column this geometry does not have; the width picks the shifted kernels.
        """

        def build(device):
            pose = self.pose.detach().to(device=device, dtype=torch.float64)
            width = 2 if self._pose_is_plain() else 4
            table = torch.zeros(self.n_angles, width, dtype=torch.float64, device=device)
            table[:, 0] = torch.cos(pose[:, 0])
            table[:, 1] = torch.sin(pose[:, 0])
            if width == 4:
                table[:, 2 : 1 + pose.shape[1]] = pose[:, 1:] * self._kernel_shift_unit
            return table.to(torch.float32).contiguous()

        return self._kernel_cached("pose", device, build)

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
        so this operator can be used inside Learned Primal-Dual networks. It is
        differentiable with respect to the pose table as well, at the cost of one
        throwaway forward: the exact adjoint hides the geometry from autograd by
        construction, so a pose that wants a gradient is differentiated through a
        forward instead. Projector buffers must have the same device and dtype as
        sinogram, as for forward().

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
        if self._geometry_requires_grad():
            # _DiscreteAdjoint hides the geometry from autograd by construction: its
            # backward is A, not d(A^T y)/d(geometry). Differentiating a throwaway
            # forward keeps both gradients, at the cost of that extra forward.
            return _vjp_adjoint(self, sinogram, create_graph=True)
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
