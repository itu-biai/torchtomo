"""Optional CUDA kernels for parallel-beam projection. Eager PyTorch stays the reference."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    _TRITON = True
except ImportError:
    triton = None
    tl = None
    _TRITON = False


def triton_kernels_available(device: torch.device) -> bool:
    return _TRITON and device.type == "cuda"


if _TRITON:

    @triton.jit
    def _forward_kernel(
        img_ptr,
        sino_ptr,
        cos_ptr,
        sin_ptr,
        B,
        S,
        A,
        pixel_size,
        stride_ib,
        stride_ih,
        stride_iw,
        stride_sb,
        stride_sa,
        stride_sd,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles = tl.cdiv(S, BLOCK_D)
        a = pid // tiles
        tile = pid % tiles
        d = tile * BLOCK_D + tl.arange(0, BLOCK_D)
        in_d = d < S
        c = (S - 1) * 0.5
        cos_a = tl.load(cos_ptr + a)
        sin_a = tl.load(sin_ptr + a)
        u = d.to(tl.float32) - c
        n_h = tl.cdiv(S, BLOCK_H)
        for b in range(B):
            acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
            img_b = img_ptr + b * stride_ib
            for ht in range(n_h):
                h = ht * BLOCK_H + tl.arange(0, BLOCK_H)
                in_h = h < S
                v = h.to(tl.float32) - c
                px = c + cos_a * u[None, :] + sin_a * v[:, None]
                py = c + cos_a * v[:, None] - sin_a * u[None, :]
                x0 = tl.floor(px)
                y0 = tl.floor(py)
                dx = px - x0
                dy = py - y0
                ix0 = x0.to(tl.int32)
                iy0 = y0.to(tl.int32)
                take = in_d[None, :] & in_h[:, None]
                m00 = take & (iy0 >= 0) & (iy0 < S) & (ix0 >= 0) & (ix0 < S)
                m10 = take & (iy0 >= 0) & (iy0 < S) & (ix0 + 1 >= 0) & (ix0 + 1 < S)
                m01 = take & (iy0 + 1 >= 0) & (iy0 + 1 < S) & (ix0 >= 0) & (ix0 < S)
                m11 = take & (iy0 + 1 >= 0) & (iy0 + 1 < S) & (ix0 + 1 >= 0) & (ix0 + 1 < S)
                v00 = tl.load(img_b + iy0 * stride_ih + ix0 * stride_iw, mask=m00, other=0.0)
                v10 = tl.load(img_b + iy0 * stride_ih + (ix0 + 1) * stride_iw, mask=m10, other=0.0)
                v01 = tl.load(img_b + (iy0 + 1) * stride_ih + ix0 * stride_iw, mask=m01, other=0.0)
                v11 = tl.load(img_b + (iy0 + 1) * stride_ih + (ix0 + 1) * stride_iw, mask=m11, other=0.0)
                sample = v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy) + v01 * (1 - dx) * dy + v11 * dx * dy
                acc += tl.sum(sample, axis=0)
            tl.store(sino_ptr + b * stride_sb + a * stride_sa + d * stride_sd, acc * pixel_size, mask=in_d)

    @triton.jit
    def _adjoint_kernel(
        sino_ptr,
        img_ptr,
        cos_ptr,
        sin_ptr,
        B,
        S,
        A,
        pixel_size,
        stride_sb,
        stride_sa,
        stride_sd,
        stride_ib,
        stride_ih,
        stride_iw,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles = tl.cdiv(S, BLOCK)
        row_tile = pid // tiles
        col_tile = pid % tiles
        i = row_tile * BLOCK + tl.arange(0, BLOCK)[:, None]
        j = col_tile * BLOCK + tl.arange(0, BLOCK)[None, :]
        in_pix = (i < S) & (j < S)
        c = (S - 1) * 0.5
        sqrt2 = 1.41421356237
        for b in range(B):
            acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
            for a in range(A):
                cos_a = tl.load(cos_ptr + a)
                sin_a = tl.load(sin_ptr + a)
                u_star = cos_a * (j - c) - sin_a * (i - c)
                v_star = sin_a * (j - c) + cos_a * (i - c)
                w0 = tl.floor(u_star + c)
                h0 = tl.floor(v_star + c)
                for dh in tl.static_range(-1, 3):
                    for dw in tl.static_range(-1, 3):
                        h = h0 + dh
                        w = w0 + dw
                        u = w - c
                        v = h - c
                        px = c + cos_a * u + sin_a * v
                        py = c + cos_a * v - sin_a * u
                        wx = 1 - tl.abs(px - j)
                        wy = 1 - tl.abs(py - i)
                        tent = tl.maximum(wx, 0.0) * tl.maximum(wy, 0.0)
                        in_lat = (w >= 0) & (w < S) & (h >= 0) & (h < S)
                        near = (tl.abs(u - u_star) < sqrt2) & (tl.abs(v - v_star) < sqrt2)
                        take = in_pix & in_lat & near & (tent > 0)
                        yv = tl.load(
                            sino_ptr + b * stride_sb + a * stride_sa + w.to(tl.int32) * stride_sd, mask=take, other=0.0
                        )
                        acc += yv * tent * pixel_size
            tl.store(img_ptr + b * stride_ib + i * stride_ih + j * stride_iw, acc, mask=in_pix)

    @triton.jit
    def _backproject_kernel(
        sino_ptr,
        img_ptr,
        cos_ptr,
        sin_ptr,
        B,
        S,
        A,
        angle_step,
        stride_sb,
        stride_sa,
        stride_sd,
        stride_ib,
        stride_ih,
        stride_iw,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles = tl.cdiv(S, BLOCK)
        row_tile = pid // tiles
        col_tile = pid % tiles
        i = row_tile * BLOCK + tl.arange(0, BLOCK)[:, None]
        j = col_tile * BLOCK + tl.arange(0, BLOCK)[None, :]
        in_pix = (i < S) & (j < S)
        c = (S - 1) * 0.5
        nx = (j - c) / c
        ny = (i - c) / c
        for b in range(B):
            acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
            for a in range(A):
                cos_a = tl.load(cos_ptr + a)
                sin_a = tl.load(sin_ptr + a)
                gx = nx * cos_a - ny * sin_a
                det = (gx + 1) * c
                x0 = tl.floor(det)
                dx = det - x0
                ix0 = x0.to(tl.int32)
                inside0 = in_pix & (ix0 >= 0) & (ix0 < S)
                inside1 = in_pix & (ix0 + 1 >= 0) & (ix0 + 1 < S)
                s0 = tl.load(sino_ptr + b * stride_sb + a * stride_sa + ix0 * stride_sd, mask=inside0, other=0.0)
                s1 = tl.load(sino_ptr + b * stride_sb + a * stride_sa + (ix0 + 1) * stride_sd, mask=inside1, other=0.0)
                acc += s0 * (1 - dx) + s1 * dx
            tl.store(img_ptr + b * stride_ib + i * stride_ih + j * stride_iw, acc * angle_step, mask=in_pix)


def _trig(angles: torch.Tensor):
    return torch.cos(angles).contiguous(), torch.sin(angles).contiguous()


def triton_forward(image: torch.Tensor, angles: torch.Tensor, pixel_size: float) -> torch.Tensor:
    batch, _, size, _ = image.shape
    n_angles = angles.numel()
    sinogram = torch.zeros(batch, 1, n_angles, size, device=image.device, dtype=image.dtype)
    cos_a, sin_a = _trig(angles)
    block_d = 32 if size >= 32 else 16
    block_h = 16 if size >= 16 else 8
    grid = (n_angles * math.ceil(size / block_d),)
    _forward_kernel[grid](
        image.reshape(batch, size, size),
        sinogram.reshape(batch, n_angles, size),
        cos_a,
        sin_a,
        batch,
        size,
        n_angles,
        float(pixel_size),
        size * size,
        size,
        1,
        n_angles * size,
        size,
        1,
        BLOCK_D=block_d,
        BLOCK_H=block_h,
    )
    return sinogram


def triton_adjoint(sinogram: torch.Tensor, angles: torch.Tensor, pixel_size: float) -> torch.Tensor:
    batch, _, n_angles, size = sinogram.shape
    image = torch.zeros(batch, 1, size, size, device=sinogram.device, dtype=sinogram.dtype)
    cos_a, sin_a = _trig(angles)
    block = 16 if size >= 16 else 8
    tiles = math.ceil(size / block)
    grid = (tiles * tiles,)
    _adjoint_kernel[grid](
        sinogram.reshape(batch, n_angles, size),
        image.reshape(batch, size, size),
        cos_a,
        sin_a,
        batch,
        size,
        n_angles,
        float(pixel_size),
        n_angles * size,
        size,
        1,
        size * size,
        size,
        1,
        BLOCK=block,
    )
    return image


def triton_backproject(sinogram: torch.Tensor, angles: torch.Tensor, angle_step: float) -> torch.Tensor:
    batch, _, n_angles, size = sinogram.shape
    image = torch.zeros(batch, 1, size, size, device=sinogram.device, dtype=sinogram.dtype)
    cos_a, sin_a = _trig(angles)
    block = 16 if size >= 16 else 8
    tiles = math.ceil(size / block)
    grid = (tiles * tiles,)
    _backproject_kernel[grid](
        sinogram.reshape(batch, n_angles, size),
        image.reshape(batch, size, size),
        cos_a,
        sin_a,
        batch,
        size,
        n_angles,
        float(angle_step),
        n_angles * size,
        size,
        1,
        size * size,
        size,
        1,
        BLOCK=block,
    )
    return image
