#!/usr/bin/env python
"""Benchmark torchtomo accuracy against scikit-image reference implementation."""

import numpy as np
import torch
from skimage.data import shepp_logan_phantom
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
)
from skimage.metrics import (
    structural_similarity as ssim,
)
from skimage.transform import iradon, radon

from torchtomo import ParallelBeam


def make_smooth_gaussian(size=256):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    phantom = np.exp(-(X**2 + Y**2) / (2 * 0.3**2))
    return phantom.astype(np.float32)


def make_sharp_circle(size=256):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    phantom = ((X**2 + Y**2) < 0.5**2).astype(np.float32)
    return phantom.astype(np.float32)


def make_shepp_logan(size=256):
    phantom = shepp_logan_phantom()
    if phantom.shape[0] != size:
        from skimage.transform import resize

        phantom = resize(phantom, (size, size), anti_aliasing=True)
    phantom = phantom / phantom.max()
    return phantom.astype(np.float32)


def benchmark_skimage(phantom, n_angles):
    theta = np.linspace(0, 180, n_angles, endpoint=False)
    sinogram = radon(phantom, theta=theta)
    recon = iradon(sinogram, theta=theta, filter_name="ramp")
    recon_clipped = np.clip(recon, 0, 1)

    psnr_val = psnr(phantom, recon_clipped, data_range=1.0)
    ssim_val = ssim(phantom, recon_clipped, data_range=1.0)
    return psnr_val, ssim_val, recon_clipped


def benchmark_torchtomo(phantom, n_angles):
    size = phantom.shape[0]
    phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

    projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
    sinogram = projector.forward(phantom_t)
    recon = projector.fbp(sinogram).clamp(0, 1)
    recon_np = recon.squeeze().numpy()

    psnr_val = psnr(phantom, recon_np, data_range=1.0)
    ssim_val = ssim(phantom, recon_np, data_range=1.0)
    return psnr_val, ssim_val, recon_np


def main():
    print("=" * 70)
    print("TorchTomo Accuracy Benchmark vs Scikit-Image Reference")
    print("=" * 70)

    phantoms = [
        ("Smooth Gaussian", make_smooth_gaussian(256)),
        ("Sharp Circle", make_sharp_circle(256)),
        ("Shepp-Logan", make_shepp_logan(256)),
    ]

    angle_configs = [180, 360, 720]

    for phantom_name, phantom in phantoms:
        print(f"\n{phantom_name}:")
        print("-" * 70)
        header = (
            f"{'Angles':>8} | {'skimage PSNR':>12} | {'torch PSNR':>12}"
            f" | {'Gap':>8} | {'skimage SSIM':>12} | {'torch SSIM':>12}"
        )
        print(header)
        print("-" * 70)

        for n_angles in angle_configs:
            psnr_sk, ssim_sk, _ = benchmark_skimage(phantom, n_angles)
            psnr_tt, ssim_tt, _ = benchmark_torchtomo(phantom, n_angles)
            gap = psnr_tt - psnr_sk

            row = (
                f"{n_angles:>8} | {psnr_sk:>10.2f} dB | {psnr_tt:>10.2f} dB"
                f" | {gap:>+7.2f} | {ssim_sk:>12.4f} | {ssim_tt:>12.4f}"
            )
            print(row)

    print("\n" + "=" * 70)
    print("Target: TorchTomo should match scikit-image within 1-2 dB")
    print("=" * 70)


if __name__ == "__main__":
    main()
