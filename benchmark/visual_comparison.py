"""Visual comparison of torchtomo vs scikit-image reconstructions at 512x512."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import iradon, radon

from torchtomo import ParallelBeam, shepp_logan


def make_disc_phantom(size):
    x = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, x)
    phantom = np.zeros((size, size), dtype=np.float32)
    phantom[X**2 + Y**2 < 0.3**2] = 1.0
    return phantom


def make_gaussian_phantom(size):
    x = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, x)
    phantom = np.exp(-(X**2 + Y**2) / (2 * 0.2**2)).astype(np.float32)
    phantom[X**2 + Y**2 > 0.95**2] = 0
    return phantom


def shepp_logan_np(size):
    return shepp_logan(size).squeeze().numpy()


def run_comparison(phantom, phantom_name, size, n_angles):
    """Run forward projection and FBP for both libraries, return metrics."""
    theta_deg = np.linspace(0, 180, n_angles, endpoint=False)

    # scikit-image
    sino_sk = radon(phantom, theta=theta_deg)
    recon_sk = np.clip(iradon(sino_sk, theta=theta_deg, filter_name="ramp"), 0, 1)

    # torchtomo
    phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
    projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
    sino_tt = projector.forward(phantom_t).squeeze().numpy()
    recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1).squeeze().numpy()

    # Metrics
    psnr_sk = psnr(phantom, recon_sk, data_range=1.0)
    psnr_tt = psnr(phantom, recon_tt, data_range=1.0)
    ssim_sk = ssim(phantom, recon_sk, data_range=1.0)
    ssim_tt = ssim(phantom, recon_tt, data_range=1.0)
    recon_psnr = psnr(recon_sk, recon_tt, data_range=1.0)
    recon_ssim = ssim(recon_sk, recon_tt, data_range=1.0)

    return {
        "sino_sk": sino_sk,
        "sino_tt": sino_tt,
        "recon_sk": recon_sk,
        "recon_tt": recon_tt,
        "psnr_sk": psnr_sk,
        "psnr_tt": psnr_tt,
        "ssim_sk": ssim_sk,
        "ssim_tt": ssim_tt,
        "recon_psnr": recon_psnr,
        "recon_ssim": recon_ssim,
    }


def plot_comparison(phantom, results, phantom_name, n_angles, save_path):
    """Create a visual comparison figure."""
    r = results
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        f"{phantom_name} phantom (512x512, {n_angles} angles)",
        fontsize=16,
        fontweight="bold",
    )

    # Row 1: Phantom, sinograms, difference
    axes[0, 0].imshow(phantom, cmap="gray")
    axes[0, 0].set_title("Original Phantom")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(r["sino_sk"], cmap="gray", aspect="auto")
    axes[0, 1].set_title("Sinogram (scikit-image)")
    axes[0, 1].axis("off")

    # Transpose torchtomo sinogram to match scikit-image orientation
    sino_tt_t = r["sino_tt"].T
    axes[0, 2].imshow(sino_tt_t * (512 / 2), cmap="gray", aspect="auto")
    axes[0, 2].set_title("Sinogram (torchtomo, scaled)")
    axes[0, 2].axis("off")

    sino_diff = np.abs(r["sino_sk"] - sino_tt_t * (512 / 2))
    im = axes[0, 3].imshow(sino_diff, cmap="hot", aspect="auto")
    axes[0, 3].set_title("Sinogram |Difference|")
    axes[0, 3].axis("off")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    # Row 2: Reconstructions and difference
    axes[1, 0].imshow(r["recon_sk"], cmap="gray")
    axes[1, 0].set_title(f"FBP (scikit-image)\nPSNR={r['psnr_sk']:.2f} dB, SSIM={r['ssim_sk']:.4f}")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(r["recon_tt"], cmap="gray")
    axes[1, 1].set_title(f"FBP (torchtomo)\nPSNR={r['psnr_tt']:.2f} dB, SSIM={r['ssim_tt']:.4f}")
    axes[1, 1].axis("off")

    recon_diff = np.abs(r["recon_sk"] - r["recon_tt"])
    im2 = axes[1, 2].imshow(recon_diff, cmap="hot")
    axes[1, 2].set_title(f"|Recon Difference|\nPSNR={r['recon_psnr']:.2f} dB, SSIM={r['recon_ssim']:.4f}")
    axes[1, 2].axis("off")
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)

    # Error vs phantom for both
    err_sk = np.abs(phantom - r["recon_sk"])
    err_tt = np.abs(phantom - r["recon_tt"])
    vmax = max(err_sk.max(), err_tt.max())
    im3 = axes[1, 3].imshow(err_tt - err_sk, cmap="RdBu_r", vmin=-vmax / 2, vmax=vmax / 2)
    axes[1, 3].set_title("Error diff (torchtomo - skimage)\nRed=torchtomo worse")
    axes[1, 3].axis("off")
    plt.colorbar(im3, ax=axes[1, 3], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    size = 512
    phantoms = {
        "disc": make_disc_phantom(size),
        "shepp-logan": shepp_logan_np(size),
        "gaussian": make_gaussian_phantom(size),
    }
    angle_counts = [180, 360, 1000]

    print("=" * 80)
    print("VISUAL COMPARISON: torchtomo vs scikit-image (512x512)")
    print("=" * 80)

    for phantom_name, phantom in phantoms.items():
        for n_angles in angle_counts:
            print(f"\n--- {phantom_name} phantom, {n_angles} angles ---")
            results = run_comparison(phantom, phantom_name, size, n_angles)
            psnr_sk = results["psnr_sk"]
            ssim_sk = results["ssim_sk"]
            psnr_tt = results["psnr_tt"]
            ssim_tt = results["ssim_tt"]
            gap = abs(psnr_tt - psnr_sk)
            recon_psnr = results["recon_psnr"]
            recon_ssim = results["recon_ssim"]
            print(f"  scikit-image:  PSNR={psnr_sk:.2f} dB, SSIM={ssim_sk:.4f}")
            print(f"  torchtomo:     PSNR={psnr_tt:.2f} dB, SSIM={ssim_tt:.4f}")
            print(f"  PSNR gap:      {gap:.2f} dB")
            print(f"  Cross-recon:   PSNR={recon_psnr:.2f} dB, SSIM={recon_ssim:.4f}")

            save_path = f"/home/user/torchtomo/tests/visual_{phantom_name}_{n_angles}angles.png"
            plot_comparison(phantom, results, phantom_name, n_angles, save_path)

    print("\n" + "=" * 80)
    print("All visual comparisons saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
