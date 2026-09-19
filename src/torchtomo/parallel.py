"""Parallel beam CT projector."""

from typing import Optional

import numpy as np
import torch

from ._sampling import grid_sample_input_backward, sample_bilinear
from ._triton_kernels import triton_adjoint, triton_backproject, triton_forward, triton_kernels_available
from .base import BaseProjector
from .filters import FilterType, apply_filter


class _TritonProject(torch.autograd.Function):
    """A x via the fused kernel; backward is the gather adjoint, including the circle mask."""

    @staticmethod
    def forward(ctx, image: torch.Tensor, projector: "ParallelBeam") -> torch.Tensor:
        ctx.projector = projector
        x = image
        if projector.circle:
            x = x * projector.circle_mask.view(1, 1, projector.img_size, projector.img_size)
        return triton_forward(x, projector.angles, projector.pixel_size)

    @staticmethod
    def backward(ctx, grad_sinogram: torch.Tensor):
        projector = ctx.projector
        grad_image = triton_adjoint(grad_sinogram.contiguous(), projector.angles, projector.pixel_size)
        if projector.circle:
            grad_image = grad_image * projector.circle_mask.view(1, 1, projector.img_size, projector.img_size)
        return grad_image, None


class _TritonBackproject(torch.autograd.Function):
    """Pixel-driven FBP backprojection via the fused kernel; VJP uses the eager interpolator."""

    @staticmethod
    def forward(ctx, sinogram: torch.Tensor, projector: "ParallelBeam") -> torch.Tensor:
        ctx.projector = projector
        ctx.sino_shape = tuple(sinogram.shape)
        recon = triton_backproject(sinogram, projector.angles, projector.angle_step)
        if projector.circle:
            recon = recon * projector.circle_mask.view(1, 1, projector.img_size, projector.img_size)
        return recon

    @staticmethod
    def backward(ctx, grad_image: torch.Tensor):
        projector = ctx.projector
        flag = projector.triton
        projector.triton = False
        try:
            with torch.inference_mode(False), torch.enable_grad():
                sino = torch.zeros(ctx.sino_shape, device=grad_image.device, dtype=grad_image.dtype, requires_grad=True)
                recon = projector.backproject(sino)
                grad_sino = torch.autograd.grad(recon, sino, grad_image.contiguous())[0]
        finally:
            projector.triton = flag
        return grad_sino, None


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
        grid_cache_bytes: int = 256 << 20,
        sparse_adjoint: bool = False,
        triton: bool = False,
        angles: Optional[torch.Tensor] = None,
    ):
        """
        Initialize parallel beam projector.

        Args:
            img_size: Image size (assumed square)
            n_angles: Number of projection angles
            n_det: Number of detector elements (default: img_size)
            angle_range: Range of angles in radians (default: 0 to pi)
            circle: If True, mask image to inscribed circle
            grid_cache_bytes: How much of the per-angle sampling grids to keep.
                Small geometries fit entirely and are then as fast as precomputing
                them; large ones exceed the budget and are rebuilt per chunk, which
                is what keeps a 512 px, 360 angle projector at megabytes instead of
                the 1.5 GB the full grids would take. Set to 0 to always rebuild.
                The default 256 MB covers 512 px with 90 angles (189 MB of forward
                grids) but not 512 px with 360 (755 MB), where about two thirds are
                rebuilt on every call and grid building is about 35% of forward
                time. Raise the budget above the grid size if that memory is free;
                do not raise the default, it comes out of LPD's headroom on an
                11 GB card.
            sparse_adjoint: If True, build a CSR matrix of the forward map and
                apply its transpose with a sparse-dense product. Off by default:
                the matrix is hundreds of megabytes at 512 px / 90 angles and
                does not fit comfortably at 512 px / 360 angles on an 11 GB card.
            triton: If True and CUDA plus Triton are available, use fused kernels
                for forward, adjoint, and analytical backprojection. Off by default.
                The eager path remains the reference and the fallback.
            angles: Explicit angle samples in radians. When omitted, n_angles
                samples cover [start, end) with spacing (end - start) / n_angles.
        """
        n_det = n_det or img_size
        super().__init__(img_size, n_angles, n_det, angle_range, angles=angles)

        self.circle = circle

        # Pixel size (assuming image spans [-1, 1])
        self.pixel_size = 2.0 / img_size

        # The per-angle sampling grids are two multiplies and an add away from these,
        # and materialising them costs [n_angles, H, W, 2] floats twice over: 1.5 GB at
        # 512 px and 360 angles. They are rebuilt per chunk instead.
        self.grid_cache_bytes = grid_cache_bytes
        self.sparse_adjoint = sparse_adjoint
        self.triton = triton
        self._sparse_adjoint_matrix = None
        self._grid_cache: dict = {}
        self._grid_cache_used = 0
        self._set_coordinate_buffers(self.angles.device, self.angles.dtype)

    def _use_triton(self, tensor: torch.Tensor) -> bool:
        return bool(self.triton) and triton_kernels_available(tensor.device, tensor.dtype)

    def _set_coordinate_buffers(self, device: torch.device, dtype: torch.dtype) -> None:
        """Rebuild the base lattice in `dtype` so `.double()` is not a float32 cast."""
        coords = torch.linspace(-1, 1, self.img_size, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("grid_x", grid_x.contiguous())
        self.register_buffer("grid_y", grid_y.contiguous())
        if self.circle:
            mask = (grid_x**2 + grid_y**2 <= 1).to(dtype=dtype)
            self.register_buffer("circle_mask", mask)

    def _cached_grid(self, kind: str, start: int, end: int, build) -> torch.Tensor:
        """Reuse a chunk's grid when the budget allows, otherwise rebuild it.

        The geometry never changes, so a cached chunk is always valid for the device
        and dtype it was built on. Anything that does not fit is simply not cached,
        which keeps the memory bounded by the budget rather than by the geometry.
        """
        key = (kind, start, end, self.grid_x.device, self.grid_x.dtype)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        grid = build()
        cost = grid.numel() * grid.element_size()
        if self._grid_cache_used + cost <= self.grid_cache_bytes:
            self._grid_cache[key] = grid
            self._grid_cache_used += cost
        return grid

    def _coordinate_pair(self, count: int) -> torch.Tensor:
        """Scratch shaped [2, count, H, W], the two coordinate planes kept contiguous.

        Building each plane in its own contiguous block and interleaving once at the
        end is twice as fast as stacking expression results, which writes and reads
        back a temporary per operation.
        """
        return torch.empty(2, count, self.img_size, self.img_size, device=self.grid_x.device, dtype=self.grid_x.dtype)

    def _forward_grid(self, start: int, end: int) -> torch.Tensor:
        """Rotation sampling grids for one chunk of angles, shape [count, H, W, 2]."""
        return self._cached_grid("forward", start, end, lambda: self._build_forward_grid(start, end))

    def _build_forward_grid(self, start: int, end: int) -> torch.Tensor:
        angles = self.angles[start:end].view(-1, 1, 1)
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        planes = self._coordinate_pair(end - start)
        torch.mul(self.grid_x, cos_a, out=planes[0])
        planes[0].addcmul_(self.grid_y, sin_a)
        torch.mul(self.grid_y, cos_a, out=planes[1])
        planes[1].addcmul_(self.grid_x, -sin_a)
        return planes.permute(1, 2, 3, 0).contiguous()

    def _backward_grid(self, start: int, end: int) -> torch.Tensor:
        """Detector lookup grids for one chunk of angles, shape [count, H, W, 2]."""
        return self._cached_grid("backward", start, end, lambda: self._build_backward_grid(start, end))

    def _build_backward_grid(self, start: int, end: int) -> torch.Tensor:
        angles = self.angles[start:end].view(-1, 1, 1)
        planes = self._coordinate_pair(end - start)
        torch.mul(self.grid_x, torch.cos(angles), out=planes[0])
        planes[0].addcmul_(self.grid_y, -torch.sin(angles))
        planes[1].zero_()
        return planes.permute(1, 2, 3, 0).contiguous()

    def _apply(self, fn, *args, **kwargs):
        # A cached grid belongs to the device and dtype it was built on.
        self._grid_cache = {}
        self._grid_cache_used = 0
        self._sparse_adjoint_matrix = None
        result = super()._apply(fn, *args, **kwargs)
        self._set_coordinate_buffers(self.angles.device, self.angles.dtype)
        return result

    def _csr_chunk(self, start: int, end: int) -> torch.Tensor:
        """Forward-map CSR for one angle chunk, rows (end-start)*n_det, from `_forward_grid`."""
        size = self.img_size
        n_det = self.n_det
        grid = self._forward_grid(start, end)
        angle_count, height, width, _ = grid.shape
        gx, gy = grid.unbind(-1)
        xs = (gx + 1) * (size - 1) / 2
        ys = (gy + 1) * (size - 1) / 2
        x0 = xs.floor().long()
        y0 = ys.floor().long()
        dx = xs - x0.to(xs.dtype)
        dy = ys - y0.to(ys.dtype)
        a_idx = torch.arange(angle_count, device=grid.device).view(-1, 1, 1).expand(angle_count, height, width)
        w_idx = torch.arange(width, device=grid.device).view(1, 1, -1).expand(angle_count, height, width)
        rows = a_idx * n_det + w_idx
        corners = (
            (x0, y0, (1 - dx) * (1 - dy)),
            (x0 + 1, y0, dx * (1 - dy)),
            (x0, y0 + 1, (1 - dx) * dy),
            (x0 + 1, y0 + 1, dx * dy),
        )
        row_parts, col_parts, val_parts = [], [], []
        mask = self.circle_mask if self.circle else None
        for ix, iy, weight in corners:
            valid = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
            if mask is not None:
                inside = mask[iy.clamp(0, size - 1), ix.clamp(0, size - 1)] > 0
                valid = valid & inside
            if not valid.any():
                continue
            row_parts.append(rows[valid])
            col_parts.append((iy * size + ix)[valid])
            val_parts.append((weight * self.pixel_size)[valid])
        nrows = angle_count * n_det
        ncols = size * size
        if not row_parts:
            crow = torch.zeros(nrows + 1, device=grid.device, dtype=torch.int64)
            col = torch.zeros(0, device=grid.device, dtype=torch.int64)
            val = torch.zeros(0, device=grid.device, dtype=grid.dtype)
            return torch.sparse_csr_tensor(crow, col, val, (nrows, ncols))
        coo = torch.sparse_coo_tensor(
            torch.stack([torch.cat(row_parts), torch.cat(col_parts)]),
            torch.cat(val_parts),
            (nrows, ncols),
        ).coalesce()
        return coo.to_sparse_csr()

    def _build_sparse_forward(self) -> torch.Tensor:
        """CSR of A^T, rows pixels, built from angle-major forward chunks then transposed."""
        if self.n_det != self.img_size:
            raise ValueError("sparse_adjoint requires n_det == img_size")
        chunk = min(8, self.n_angles)
        parts = [self._csr_chunk(start, min(start + chunk, self.n_angles)) for start in range(0, self.n_angles, chunk)]
        values = torch.cat([part.values() for part in parts])
        cols = torch.cat([part.col_indices() for part in parts])
        crows = []
        nnz = 0
        for part in parts:
            crow = part.crow_indices()
            crows.append(crow[:-1] + nnz)
            nnz += int(crow[-1].item())
        index_dtype = torch.int32 if nnz < (1 << 31) and self.img_size * self.img_size < (1 << 31) else torch.int64
        crow = torch.cat(crows + [torch.tensor([nnz], device=cols.device, dtype=crow.dtype)]).to(index_dtype)
        cols = cols.to(index_dtype)
        nrows = self.n_angles * self.n_det
        ncols = self.img_size * self.img_size
        forward = torch.sparse_csr_tensor(crow, cols, values, (nrows, ncols))
        # CSR @ dense is the fast product; A.t() is CSC and is as slow as dense @ CSR.
        adjoint = forward.t().to_sparse_csr()
        if index_dtype == torch.int32:
            adjoint = torch.sparse_csr_tensor(
                adjoint.crow_indices().to(torch.int32),
                adjoint.col_indices().to(torch.int32),
                adjoint.values(),
                adjoint.shape,
            )
        return adjoint

    def _ensure_sparse_adjoint_matrix(self) -> torch.Tensor:
        if self._sparse_adjoint_matrix is None:
            self._sparse_adjoint_matrix = self._build_sparse_forward()
        return self._sparse_adjoint_matrix

    def _direct_adjoint(self, sinogram: torch.Tensor) -> torch.Tensor:
        """A^T y by calling grid_sample's input backward, skipping a throwaway forward.

        Same kernel autograd would call, with the grid gradient masked off. The
        operator is linear, so the image gradient does not depend on the image.
        """
        batch = sinogram.shape[0]
        if self._use_triton(sinogram):
            out = triton_adjoint(sinogram, self.angles, self.pixel_size)
            if self.circle:
                out = out * self.circle_mask.view(1, 1, self.img_size, self.img_size)
            return out
        if self.sparse_adjoint:
            matrix = self._ensure_sparse_adjoint_matrix()
            return (matrix @ sinogram.reshape(batch, -1).t()).t().reshape(batch, 1, self.img_size, self.img_size)
        size = self.img_size
        chunk = self._angle_chunk_size(batch, size * size, sinogram.device)
        out = torch.zeros(1, batch, size, size, device=sinogram.device, dtype=sinogram.dtype)
        shape_only = torch.empty(1, batch, size, size, device=sinogram.device, dtype=sinogram.dtype)
        for start in range(0, self.n_angles, chunk):
            end = min(start + chunk, self.n_angles)
            angle_count = end - start
            # Ray samples on H, angles along W: the ray-sum adjoint is a stride-0
            # broadcast. The cached grid is angle-major (the fast forward layout),
            # so the permute copies one chunk.
            grid = self._forward_grid(start, end).permute(1, 0, 2, 3).contiguous()
            grid = grid.reshape(1, size, angle_count * size, 2)
            grad = (sinogram[:, 0, start:end, :] * self.pixel_size).reshape(batch, 1, angle_count * size)
            grad = grad.expand(batch, size, angle_count * size).unsqueeze(0)
            out += grid_sample_input_backward(grad, shape_only, grid)
        out = out.view(batch, 1, size, size)
        if self.circle:
            out = out * self.circle_mask.view(1, 1, size, size)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward projection (Radon transform): image -> sinogram.

        Args:
            x: Image tensor [B, 1, H, W]

        Returns:
            Sinogram [B, 1, n_angles, n_det]
        """
        B = x.shape[0]
        if self._use_triton(x):
            return _TritonProject.apply(x, self)

        # Apply circle mask if enabled
        if self.circle:
            x = x * self.circle_mask.view(1, 1, self.img_size, self.img_size)

        projections = []
        size = self.img_size
        chunk_size = self._angle_chunk_size(B, size * size, x.device)

        # The batch rides in the channel dimension and the angles are stacked along the
        # sampled height, so one grid serves every image and the image is never copied.
        # grid_sample applies the same grid to all channels, which is exactly what a
        # geometry shared across a batch needs.
        for start in range(0, self.n_angles, chunk_size):
            end = min(start + chunk_size, self.n_angles)
            angle_count = end - start
            grid = self._forward_grid(start, end).reshape(1, angle_count * size, size, 2)

            rotated = sample_bilinear(x.reshape(1, B, size, size), grid)

            projection = rotated.view(B, angle_count, size, size).sum(dim=2) * self.pixel_size
            projections.append(projection.unsqueeze(1))

        # Stack to sinogram [B, 1, n_angles, n_det]
        sinogram = torch.cat(projections, dim=2)

        return sinogram

    def backproject(self, sinogram: torch.Tensor) -> torch.Tensor:
        """
        Analytical backprojection for FBP: sinogram -> image.

        Smears projections across the image and applies angular normalization.
        This is not the transpose of forward()'s discrete interpolation.
        Use backward() or adjoint() for a matched discrete operator pair.

        Args:
            sinogram: Sinogram [B, 1, n_angles, n_det]

        Returns:
            Back-projected image [B, 1, H, W]
        """
        B = sinogram.shape[0]
        if self._use_triton(sinogram):
            return _TritonBackproject.apply(sinogram, self)
        recon = torch.zeros(B, 1, self.img_size, self.img_size, device=sinogram.device)
        chunk_size = self._angle_chunk_size(B, self.img_size * self.img_size, sinogram.device)

        for start in range(0, self.n_angles, chunk_size):
            end = min(start + chunk_size, self.n_angles)
            angle_count = end - start

            # Angles lead the batch here, because each angle samples its own detector
            # row: that keeps one grid per angle instead of one per angle and image,
            # and the only copy is the detector rows, which are tiny beside the grids.
            sino_rows = sinogram[:, :, start:end, :].reshape(B, angle_count, self.n_det)
            sino_rows = sino_rows.permute(1, 0, 2).reshape(angle_count, B, 1, self.n_det)
            grid = self._backward_grid(start, end)

            contribution = sample_bilinear(sino_rows, grid)
            recon += contribution.sum(dim=0).unsqueeze(1)

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
        recon = self.backproject(filtered_sino)

        return recon
