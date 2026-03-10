#!/usr/bin/env python
"""Generate visualization of torchtomo reconstruction quality."""

import matplotlib.pyplot as plt
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
)
from skimage.metrics import (
    structural_similarity as ssim,
)

from torchtomo import FanBeam, ParallelBeam, shepp_logan


def compute_metrics(recon, phantom):
    recon_clipped = recon.clamp(0, 1)
    phantom_np = phantom.squeeze().numpy()
    recon_np = recon_clipped.squeeze().numpy()
    mse = ((recon_clipped - phantom) ** 2).mean().item()
    psnr_val = psnr(phantom_np, recon_np, data_range=1.0)
    ssim_val = ssim(phantom_np, recon_np, data_range=1.0)
    return mse, psnr_val, ssim_val


def main():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Parallel Beam
    projector = ParallelBeam(img_size=256, n_angles=180, n_det=256)
    phantom = shepp_logan(256)
    sinogram = projector.forward(phantom)
    recon = projector.fbp(sinogram)

    mse_parallel, psnr_parallel, ssim_parallel = compute_metrics(recon, phantom)

    axes[0, 0].imshow(phantom.squeeze().numpy(), cmap="gray")
    axes[0, 0].set_title("Original Phantom")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(sinogram.squeeze().numpy(), cmap="gray", aspect="auto")
    axes[0, 1].set_title("Parallel Beam Sinogram")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(recon.squeeze().numpy(), cmap="gray")
    axes[0, 2].set_title(
        f"FBP Reconstruction\nMSE: {mse_parallel:.6f} | PSNR: {psnr_parallel:.1f} dB | SSIM: {ssim_parallel:.3f}"
    )
    axes[0, 2].axis("off")

    error = (recon - phantom).abs()
    im = axes[0, 3].imshow(error.squeeze().numpy(), cmap="hot", vmin=0, vmax=0.3)
    axes[0, 3].set_title("Absolute Error")
    axes[0, 3].axis("off")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    # Fan Beam
    projector = FanBeam(
        img_size=256,
        n_angles=360,
        n_det=400,
        src_dist=500,
        det_dist=500,
    )
    sinogram = projector.forward(phantom)
    recon = projector.fbp(sinogram)

    mse_fan, psnr_fan, ssim_fan = compute_metrics(recon, phantom)

    axes[1, 0].imshow(phantom.squeeze().numpy(), cmap="gray")
    axes[1, 0].set_title("Original Phantom")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(sinogram.squeeze().numpy(), cmap="gray", aspect="auto")
    axes[1, 1].set_title("Fan Beam Sinogram")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(recon.squeeze().numpy(), cmap="gray")
    axes[1, 2].set_title(f"FBP Reconstruction\nMSE: {mse_fan:.6f} | PSNR: {psnr_fan:.1f} dB | SSIM: {ssim_fan:.3f}")
    axes[1, 2].axis("off")

    error = (recon - phantom).abs()
    im = axes[1, 3].imshow(error.squeeze().numpy(), cmap="hot", vmin=0, vmax=0.3)
    axes[1, 3].set_title("Absolute Error")
    axes[1, 3].axis("off")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046)

    # Row labels
    axes[0, 0].text(
        -0.15,
        0.5,
        "Parallel\nBeam",
        transform=axes[0, 0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="center",
        ha="center",
    )
    axes[1, 0].text(
        -0.15,
        0.5,
        "Fan\nBeam",
        transform=axes[1, 0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="center",
        ha="center",
    )

    plt.suptitle("TorchTomo - Differentiable CT Reconstruction", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("current.png", dpi=150, bbox_inches="tight")
    print("Saved current.png")
    print()
    print("Parallel Beam:")
    print(f"  MSE: {mse_parallel:.6f} | PSNR: {psnr_parallel:.2f} dB | SSIM: {ssim_parallel:.4f}")
    print("Fan Beam:")
    print(f"  MSE: {mse_fan:.6f} | PSNR: {psnr_fan:.2f} dB | SSIM: {ssim_fan:.4f}")


if __name__ == "__main__":
    main()
