"""Fan beam CT projector with flat detector."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from . import _cuda_kernels
from ._sampling import grid_sample_input_backward, sample_bilinear
from .base import BaseProjector, _check_backend, _KernelBackproject, _KernelGeometryProject, _KernelProject
from .filters import FilterType, apply_filter

# Grids only the PyTorch path reads: about 2.2 GB at 512 px and 360 angles.
_EAGER_GEOMETRY = ("ray_grids", "ray_lengths", "backward_grids", "backward_weights")


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

    # Both shifts are lateral, perpendicular to the source-to-detector axis.
    _POSE_COLUMNS = ("angle", "detector_shift", "source_shift")

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
        backend: str = "torch",
        approximate: bool = False,
        learnable_geometry: bool = False,
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
            backend: "torch" (default) runs everywhere on PyTorch operations.
                "cuda" runs float32 CUDA tensors on kernels compiled at first use
                by the NVRTC that ships with PyTorch; the forward and adjoint are
                an exact matched pair of their own, and it keeps only per-ray
                tables on the device. "auto" is "cuda" where NVRTC loads and
                "torch" everywhere else, decided once here: `projector.backend`
                reports which it became. Either way the PyTorch path's sampling
                grids (about 2.2 GB at 512 px, 360 angles) are built only if that
                path is used, e.g. on the CPU or in float64.
            approximate: With backend="cuda" or "auto", sample the image through
                the GPU's texture units (hardware bilinear interpolation with 8-bit weights,
                about 3e-4 relative error) and use a pixel-driven adjoint (linear
                interpolation on the detector with the fan's ray-spacing weight).
                Faster, but the forward and adjoint are no longer each other's
                exact transpose. Off by default.
            learnable_geometry: Register the pose table, [n_angles, 3] of angle and
                per-view detector and source shift in pixels, and `distances`, [3]
                of src_dist, det_dist and det_width in pixels, as nn.Parameters so
                an optimiser reaches them through .parameters(). Off by default,
                where both are ordinary buffers and `requires_grad_(True)` on
                either still gives a one-off gradient. With backend="cuda", forward() and
                adjoint() differentiate the geometry on the kernels, to first
                order; everything else, and every order, on the PyTorch path.
        """
        src_dist = float(2 * img_size if src_dist is None else src_dist)
        det_dist = float(2 * img_size if det_dist is None else det_dist)
        n_det = int(round(1.5 * img_size) if n_det is None else n_det)
        n_samples = int(img_size if n_samples is None else n_samples)
        super().__init__(img_size, n_angles, n_det, angle_range, angles=angles, learnable_geometry=learnable_geometry)
        self.backend = _check_backend(backend, ("auto", "torch", "cuda"))
        if approximate and backend not in ("cuda", "auto"):
            raise ValueError("approximate=True needs backend='cuda' or 'auto'")
        if approximate and n_samples < 2:
            raise ValueError("approximate=True needs at least two samples per ray")
        self.approximate = approximate

        magnification = (src_dist + det_dist) / src_dist
        if det_spacing is not None:
            det_width = det_spacing * n_det
        elif det_width is None:
            det_width = 1.5 * magnification * img_size
        self.n_samples = n_samples
        self.circle = circle

        self.scale = 2.0 / img_size
        # The distances in float64 as built or last written; `distances` is the
        # tensor a gradient reaches, in the dtype of the rest of the geometry.
        self._distance_master = (src_dist, det_dist, float(det_width))
        values = torch.tensor(self._distance_master, dtype=torch.float32)
        if learnable_geometry:
            self.register_parameter("distances", nn.Parameter(values))
        else:
            self.register_buffer("distances", values)
        self._set_geometry_buffers()

    @property
    def src_dist(self) -> float:
        """Source to isocentre in pixels; `distances[0]` is the tensor behind it."""
        return self._distance_floats()[0]

    @property
    def det_dist(self) -> float:
        """Isocentre to detector in pixels; `distances[1]` is the tensor behind it."""
        return self._distance_floats()[1]

    @property
    def det_width(self) -> float:
        """Detector width in pixels; `distances[2]` is the tensor behind it."""
        return self._distance_floats()[2]

    def set_distances(
        self, src_dist: Optional[float] = None, det_dist: Optional[float] = None, det_width: Optional[float] = None
    ) -> None:
        """Write the fan's distances in pixels, with no gradient; see set_pose."""
        values = list(self._distance_floats())
        for index, value in enumerate((src_dist, det_dist, det_width)):
            if value is not None:
                values[index] = float(value)
        self._distance_master = tuple(values)
        with torch.no_grad():
            self.distances.copy_(torch.tensor(values, dtype=self.distances.dtype))
        self._geometry_changed()

    def _distance_floats(self) -> tuple[float, float, float]:
        """The distances in pixels as floats. A geometry that can move reads its tensor."""
        if self._geometry_is_mutable():
            src, det, width = self.distances.detach().double().cpu().tolist()
            return src, det, width
        return self._distance_master

    def _lengths(self, dtype: torch.dtype, device: torch.device):
        """(source, detector, width) in normalised coordinates, for building grids.

        Floats for a fixed geometry, the arithmetic that predates learnable
        distances; tensors from `distances`, inside the graph, for one that can move.
        """
        if self._geometry_is_mutable():
            src, det, width = (self.distances.to(device=device, dtype=dtype) * self.scale).unbind()
            return src, det, width
        src, det, width = self._distance_master
        return src * self.scale, det * self.scale, width * self.scale

    def _length_floats(self) -> tuple[float, float, float]:
        """(source, detector, width) in normalised coordinates, as kernel arguments."""
        src, det, width = self._distance_floats()
        return src * self.scale, det * self.scale, width * self.scale

    def _detector_offsets(self, width, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Bin centres across the detector, -width / 2 to width / 2."""
        if isinstance(width, torch.Tensor):
            return torch.linspace(-1, 1, self.n_det, dtype=dtype, device=device) * (width / 2)
        return torch.linspace(-width / 2, width / 2, self.n_det, dtype=dtype, device=device)

    def _geometry_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.pose, self.distances)

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        if name == "distances" and isinstance(value, torch.Tensor):
            # As for the pose: a table assigned wholesale moves the geometry.
            if not value.requires_grad:
                src, det, width = value.detach().double().cpu().tolist()
                self._distance_master = (src, det, width)
            self._geometry_changed()

    def _geometry_changed(self) -> None:
        super()._geometry_changed()
        if "distances" in self._parameters or "distances" in self._buffers:
            self._set_geometry_buffers()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # 0.3 checkpoints hold no distances: they are the ones the constructor was given.
        key = prefix + "distances"
        if key not in state_dict:
            state_dict[key] = self.distances.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        src, det, width = self.distances.detach().double().cpu().tolist()
        self._distance_master = (src, det, width)
        self._set_geometry_buffers()

    def _apply(self, fn, *args, **kwargs):
        # The distances get the same treatment as the angles in BaseProjector._apply.
        master = torch.tensor(self._distance_master, dtype=self.distances.dtype, device=self.distances.device)
        as_built = torch.equal(self.distances.detach(), master)
        result = super()._apply(fn, *args, **kwargs)
        if as_built:
            with torch.no_grad():
                self.distances.copy_(torch.tensor(self._distance_master, dtype=self.distances.dtype))
        self._set_geometry_buffers()
        return result

    def _set_geometry_buffers(self) -> None:
        """Rebuild ray and backprojection grids in the current angles dtype."""
        dtype, device = self.angles.dtype, self.angles.device
        # The PyTorch path's four grids are the expensive part of this geometry and
        # a subclass that replaces every operator, or a run that stays on the CUDA
        # kernels, never touches them. They start empty and __getattr__ builds them
        # on first read.
        for name in _EAGER_GEOMETRY:
            self._set_buffer(name, None)

        src, det, width = self._length_floats()
        det_pos = torch.linspace(-width / 2, width / 2, self.n_det, dtype=dtype, device=device)
        D = src + det
        cos_weight = D / torch.sqrt(D**2 + det_pos**2)
        self.register_buffer("cos_weight", cos_weight)

        if self.circle:
            coords = torch.linspace(-1, 1, self.img_size, dtype=dtype, device=device)
            y, x = torch.meshgrid(coords, coords, indexing="ij")
            mask = (x**2 + y**2 <= 1).to(dtype=dtype)
            self.register_buffer("circle_mask", mask)

    def _build_eager_geometry(self) -> None:
        # Buffers are geometry, not activations: build them as ordinary tensors even
        # when the first use happens under inference_mode.
        with torch.inference_mode(False), torch.no_grad():
            ray_grids, ray_lengths = self._precompute_ray_grids()
            back_grids, weights = self._precompute_backward_grids()
        self._set_buffer("ray_grids", ray_grids)
        self._set_buffer("ray_lengths", ray_lengths)
        self._set_buffer("backward_grids", back_grids)
        self._set_buffer("backward_weights", weights)

    def _invalidate_geometry(self) -> None:
        # Back to None, so __getattr__ rebuilds them from the pose on the next read.
        for name in _EAGER_GEOMETRY:
            self._set_buffer(name, None)

    def _set_buffer(self, name: str, value: torch.Tensor | None) -> None:
        # register_buffer probes hasattr(), which would build a lazy buffer to replace it.
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    def __getattr__(self, name: str):
        # The PyTorch path's grids start as None and are built on first access, so
        # falling back (CPU, float64) or reading them still works.
        if name in _EAGER_GEOMETRY:
            buffers = self.__dict__.get("_buffers", {})
            if name in buffers and buffers[name] is None:
                self._build_eager_geometry()
                return buffers[name]
        return super().__getattr__(name)

    @property
    def shift_scale(self) -> float:
        """One pixel of pose offset in normalised coordinates.

        Fan beam states src_dist, det_dist and det_width in pixels of 2 / img_size,
        so a pose offset is measured in the same pixel and every length in this
        geometry means the same thing.
        """
        return self.scale

    @property
    def _kernel_shift_unit(self) -> float:
        # fan_backproject works in the eager path's normalised coordinates.
        return self.shift_scale

    @property
    def source_shift(self) -> torch.Tensor:
        """Per-view lateral source offset in pixels, column 2 of the pose table."""
        return self.pose[:, 2]

    def _kernel_tables(self, device: torch.device, differentiable: bool = False) -> tuple[torch.Tensor, ...]:
        """What the kernels read; rays and weights in the graph when `differentiable`.

        The adjoint's candidate search reads inv_steps and views too, but only to
        find the rays it then evaluates exactly, so they carry no gradient.
        """
        if differentiable:
            rays, inv_steps, weights, views = self._ray_tables(self.pose.to(device=device, dtype=torch.float64))
            return rays, inv_steps.detach(), weights, views.detach()
        return self._kernel_rays(device)

    def _kernel_table_grads(self, image: torch.Tensor, grad: torch.Tensor, tables) -> tuple:
        rays, _, weights, _ = tables
        mask = self._kernel_mask(image.device)
        grad_rays, grad_weights = _cuda_kernels.fan_ray_grad(
            image, grad, rays, weights, mask, self.n_angles, self.n_det, self.n_samples
        )
        return grad_rays, None, grad_weights, None

    def _kernel_forward(self, x: torch.Tensor, tables=None) -> torch.Tensor:
        rays, _, weights, _ = tables or self._kernel_tables(x.device)
        mask = self._kernel_mask(x.device)
        if self.approximate:
            return _cuda_kernels.fan_forward_texture(
                x, rays, weights, mask, self.n_angles, self.n_det, self.n_samples, self._kernel_cache
            )
        return _cuda_kernels.fan_forward(x, rays, weights, mask, self.n_angles, self.n_det, self.n_samples)

    def _kernel_adjoint(self, sinogram: torch.Tensor, tables=None) -> torch.Tensor:
        rays, inv_steps, weights, views = tables or self._kernel_tables(sinogram.device)
        src, det, width = self._length_floats()
        alpha = (src + det) * (self.n_det - 1) / width
        if self.approximate:
            centre = (self.img_size - 1) / 2
            return _cuda_kernels.fan_adjoint_pixel(
                sinogram,
                views,
                self._kernel_mask(sinogram.device),
                self.img_size,
                alpha,
                (self.n_det - 1) / 2,
                (src + det) * centre,
                width / (self.n_det - 1) * centre,
                (self.n_samples - 1) / (self.n_samples * centre),
                shifted=not self._pose_is_plain(),
            )
        return _cuda_kernels.fan_adjoint(
            sinogram,
            rays,
            inv_steps,
            weights,
            views,
            self._kernel_mask(sinogram.device),
            self.img_size,
            self.n_samples,
            alpha,
            (self.n_det - 1) / 2,
        )

    def _kernel_backproject(self, sinogram: torch.Tensor) -> torch.Tensor:
        device = sinogram.device
        src, det, width = self._length_floats()
        return _cuda_kernels.fan_backproject(
            sinogram,
            self._kernel_pose(device),
            self._kernel_coords(device),
            self._kernel_mask(device),
            src,
            det,
            width / 2,
            float(self.angle_step) / 2,
        )

    def _kernel_rays(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-ray tables for the CUDA kernels, built in float64 and stored in float32.

        rays [n_angles * n_det, 4]: entry point and sample step in pixel coordinates.
        inv_steps [n_angles * n_det, 2]: reciprocal of the step per axis, 0 where it is 0.
        weights [n_angles * n_det]: chord length over n_samples.
        views [n_angles * 2, 4]: source in pixels and detector direction, then the
        detector normal and the view's bin offset, (source shift - detector shift)
        over the bin pitch; the adjoint uses them to find candidate bins.
        """

        def build(device):
            return self._ray_tables(self.pose.detach().to(device=device, dtype=torch.float64))

        return self._kernel_cached("rays", device, build)

    def _ray_tables(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """_kernel_rays from a float64 pose, differentiable in it."""
        f64, device = torch.float64, pose.device
        _, _, width = self._lengths(f64, device)
        src_x, src_y, det_cx, det_cy, cos_a, sin_a = self._source_and_detector(pose, (-1, 1))
        offsets = self._detector_offsets(width, f64, device).view(1, -1)
        dir_x = det_cx + offsets * cos_a - src_x
        dir_y = det_cy + offsets * sin_a - src_y
        length = torch.sqrt(dir_x**2 + dir_y**2)
        dir_x, dir_y = dir_x / length, dir_y / length
        b = 2 * (src_x * dir_x + src_y * dir_y)
        c = src_x**2 + src_y**2 - 1.0
        root = torch.sqrt(torch.clamp(b**2 - 4 * c, min=0))
        t_entry = torch.clamp((-b - root) / 2, min=0)
        t_exit = torch.maximum((-b + root) / 2, t_entry)
        chord = t_exit - t_entry
        centre = (self.img_size - 1) / 2
        spacing = chord / (self.n_samples - 1) if self.n_samples > 1 else torch.zeros_like(chord)
        rays = torch.stack(
            (
                (src_x + t_entry * dir_x + 1) * centre,
                (src_y + t_entry * dir_y + 1) * centre,
                spacing * dir_x * centre,
                spacing * dir_y * centre,
            ),
            dim=-1,
        )
        steps = rays[..., 2:]
        moving = steps.abs() > 1e-12
        inv_steps = torch.where(moving, 1.0 / torch.where(moving, steps, torch.ones_like(steps)), 0.0)
        weights = chord / self.n_samples
        zeros = torch.zeros_like(cos_a)
        pitch = width / (self.n_det - 1)
        lateral = (pose[:, 2:3] - pose[:, 1:2]) * self.shift_scale / pitch
        views = torch.cat(
            ((src_x + 1) * centre, (src_y + 1) * centre, cos_a, sin_a, sin_a, -cos_a, lateral, zeros), dim=-1
        )
        as32 = lambda t: t.to(torch.float32).contiguous()  # noqa: E731
        return (
            as32(rays.reshape(-1, 4)),
            as32(inv_steps.reshape(-1, 2)),
            as32(weights.reshape(-1)),
            as32(views.reshape(-1, 4)),
        )

    def _source_and_detector(self, pose: torch.Tensor, shape: tuple[int, ...]):
        """Source and detector centre for these views, shaped to broadcast over `shape`.

        A lateral shift moves its end of the ray along the detector direction
        (cos, sin), which is perpendicular to the source-to-detector axis. A
        constant detector shift is a centre-of-rotation error; a per-view one is
        in-plane motion; a source shift tilts the fan.
        """
        src, det, _ = self._lengths(pose.dtype, pose.device)
        angles = pose[:, 0].view(shape)
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        src_x = -src * sin_a
        src_y = src * cos_a
        det_cx = det * sin_a
        det_cy = -det * cos_a
        if not self._pose_is_plain():
            det_u = pose[:, 1].view(shape) * self.shift_scale
            src_u = pose[:, 2].view(shape) * self.shift_scale
            src_x = src_x + src_u * cos_a
            src_y = src_y + src_u * sin_a
            det_cx = det_cx + det_u * cos_a
            det_cy = det_cy + det_u * sin_a
        return src_x, src_y, det_cx, det_cy, cos_a, sin_a

    def _ray_chunk(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Ray grids and path lengths for one chunk of views.

        A fixed geometry slices the buffers it built once; a geometry that can move
        rebuilds the chunk, which is also the only size that fits: the whole set is
        about 2.2 GB at 512 px and 360 angles.
        """
        if self._geometry_is_mutable():
            return self._precompute_ray_grids(start, end)
        return self.ray_grids[start:end], self.ray_lengths[start:end]

    def _backward_chunk(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Backprojection grids and weights for one chunk of views; see _ray_chunk."""
        if self._geometry_is_mutable():
            return self._precompute_backward_grids(start, end)
        return self.backward_grids[start:end], self.backward_weights[start:end]

    def _precompute_ray_grids(self, start: int = 0, end: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sampling grids [count, n_det, n_samples, 2] and path lengths [count, n_det]."""
        pose = self.pose[start : self.n_angles if end is None else end]
        dtype, device = pose.dtype, pose.device
        src_x, src_y, det_cx, det_cy, cos_a, sin_a = self._source_and_detector(pose, (-1, 1))
        _, _, width = self._lengths(dtype, device)
        det_offsets = self._detector_offsets(width, dtype, device).view(1, -1)
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

    def _precompute_backward_grids(
        self, start: int = 0, end: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backprojection grids [count, H, W, 2] and 1/U² weights [count, H, W]."""
        pose = self.pose[start : self.n_angles if end is None else end]
        dtype, device = pose.dtype, pose.device
        coords = torch.linspace(-1, 1, self.img_size, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        src_x, src_y, det_cx, det_cy, cos_a, sin_a = self._source_and_detector(pose, (-1, 1, 1))
        px_x = grid_x - src_x
        px_y = grid_y - src_y
        # Depth along the detector normal (sin, -cos), where the detector plane sits
        # src + det from the source whatever the lateral shifts: the line from the
        # source to a shifted detector centre is tilted and is not that normal.
        proj_len = px_x * sin_a - px_y * cos_a
        src, det, width = self._lengths(dtype, device)
        t = (src + det) / (proj_len + 1e-8)
        int_x = src_x + t * px_x
        int_y = src_y + t * px_y
        det_offset = (int_x - det_cx) * cos_a + (int_y - det_cy) * sin_a
        det_normalized = det_offset / (width / 2)
        grid = torch.stack([det_normalized, torch.zeros_like(det_normalized)], dim=-1)
        # Distance from the source plane, measured along the axis: a lateral shift
        # slides both ends sideways and leaves this weight where it was.
        U = (src + grid_x * sin_a - grid_y * cos_a) / src
        weight = 1.0 / U.clamp_min(1e-6).square()
        return grid, weight

    def _direct_adjoint(self, sinogram: torch.Tensor) -> torch.Tensor:
        """A^T y by calling grid_sample's input backward, skipping a throwaway forward."""
        batch = sinogram.shape[0]
        size = self.img_size
        chunk = self._angle_chunk_size(batch, self.n_det * self.n_samples, sinogram.device)
        out = torch.zeros(1, batch, size, size, device=sinogram.device, dtype=sinogram.dtype)
        shape_only = torch.empty(1, batch, size, size, device=sinogram.device, dtype=sinogram.dtype)
        for start in range(0, self.n_angles, chunk):
            end = min(start + chunk, self.n_angles)
            angle_count = end - start
            grids, lengths = self._ray_chunk(start, end)
            grid = grids.reshape(1, angle_count * self.n_det, self.n_samples, 2)
            scale = lengths / self.n_samples
            grad = (sinogram[:, 0, start:end, :] * scale).reshape(1, batch, angle_count * self.n_det, 1)
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
        if self._use_kernels(x):
            return _KernelProject.apply(x, self)
        if self._use_geometry_kernels(x):
            return _KernelGeometryProject.apply(x, self, *self._kernel_tables(x.device, differentiable=True))

        if self.circle:
            x = x * self.circle_mask.view(1, 1, self.img_size, self.img_size)

        size = self.img_size
        projections = []
        chunk_size = self._angle_chunk_size(B, self.n_det * self.n_samples, x.device)

        for start in range(0, self.n_angles, chunk_size):
            end = min(start + chunk_size, self.n_angles)
            angle_count = end - start
            grids, lengths = self._ray_chunk(start, end)
            grid = grids.reshape(1, angle_count * self.n_det, self.n_samples, 2)
            samples = sample_bilinear(x.reshape(1, B, size, size), grid)
            projection = samples.view(B, angle_count, self.n_det, self.n_samples).mean(dim=-1)
            projection = projection * lengths
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
        if self._use_kernels(sinogram):
            return _KernelBackproject.apply(sinogram, self)
        return self._backproject_eager(sinogram)

    def _backproject_eager(self, sinogram: torch.Tensor) -> torch.Tensor:
        B = sinogram.shape[0]
        recon = torch.zeros(B, 1, self.img_size, self.img_size, device=sinogram.device, dtype=sinogram.dtype)
        chunk_size = self._angle_chunk_size(B, self.img_size * self.img_size, sinogram.device)

        for start in range(0, self.n_angles, chunk_size):
            end = min(start + chunk_size, self.n_angles)
            angle_count = end - start
            rows = sinogram[:, 0, start:end, :].permute(1, 0, 2).reshape(angle_count, B, 1, self.n_det)
            grids, weights = self._backward_chunk(start, end)
            contrib = sample_bilinear(rows, grids)
            weight = weights.unsqueeze(1)
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
        if self._geometry_is_mutable():
            # The weights of a geometry that can move, from its distances as they are.
            src, det, width = self._lengths(sinogram.dtype, sinogram.device)
            span = src + det
            cos_w = span / torch.sqrt(span**2 + self._detector_offsets(width, sinogram.dtype, sinogram.device) ** 2)
            virt_px = width / self.scale / self.n_det / (span / src)
        else:
            cos_w = self.cos_weight
            # Ramp is built in bin units; convert to the virtual detector at the isocentre.
            mag = (self.src_dist + self.det_dist) / self.src_dist
            virt_px = self.det_width / self.n_det / mag
        weighted_sino = sinogram * cos_w.view(1, 1, 1, -1)

        filtered_sino = apply_filter(weighted_sino, filter_name)
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
