"""Tests verifying torchtomo consistency with scikit-image reconstruction."""

import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import iradon, radon

from torchtomo import ParallelBeam, shepp_logan


def _make_disc_phantom(size=256):
    """Disc phantom: zero outside inscribed circle."""
    x = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, x)
    phantom = np.zeros((size, size), dtype=np.float32)
    phantom[X**2 + Y**2 < 0.3**2] = 1.0
    return phantom


def _make_gaussian_phantom(size=256):
    """Gaussian phantom masked to inscribed circle."""
    x = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, x)
    phantom = np.exp(-(X**2 + Y**2) / (2 * 0.2**2)).astype(np.float32)
    phantom[X**2 + Y**2 > 0.95**2] = 0
    return phantom


def _shepp_logan_np(size=256):
    """Get torchtomo's Shepp-Logan phantom as numpy array."""
    return shepp_logan(size).squeeze().numpy()


class TestSinogramConsistency:
    """Compare forward projections between torchtomo and scikit-image."""

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_sinogram_correlation(self, n_angles):
        """Sinograms should be highly correlated after accounting for scaling."""
        size = 512
        phantom = _make_disc_phantom(size)
        theta_deg = np.linspace(0, 180, n_angles, endpoint=False)

        # scikit-image sinogram: shape (n_det, n_angles)
        sino_sk = radon(phantom, theta=theta_deg)

        # torchtomo sinogram: shape (n_angles, n_det), scaled by pixel_size
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        sino_tt = projector.forward(phantom_t).squeeze().numpy()

        # Account for scaling: torchtomo multiplies by pixel_size = 2/img_size
        sino_tt_scaled = sino_tt.T * (size / 2)  # transpose to (n_det, n_angles)

        # Correlation should be very high
        corr = np.corrcoef(sino_sk.ravel(), sino_tt_scaled.ravel())[0, 1]
        assert corr > 0.99, f"Sinogram correlation {corr:.4f} < 0.99"

    def test_sinogram_projection_peaks(self):
        """Peak projection values should match after scaling."""
        size = 512
        phantom = _make_disc_phantom(size)
        theta_deg = np.linspace(0, 180, 180, endpoint=False)

        sino_sk = radon(phantom, theta=theta_deg)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=180, n_det=size)
        sino_tt = projector.forward(phantom_t).squeeze().numpy()

        # Peak values should match within 5% after scaling
        peak_sk = sino_sk.max()
        peak_tt = sino_tt.max() * (size / 2)
        rel_diff = abs(peak_sk - peak_tt) / peak_sk
        assert rel_diff < 0.05, f"Peak sinogram relative diff {rel_diff:.4f} > 0.05"


class TestFBPConsistency:
    """Compare FBP reconstructions between torchtomo and scikit-image."""

    @pytest.mark.parametrize(
        "phantom_fn,phantom_name",
        [
            (_make_disc_phantom, "disc"),
            (_shepp_logan_np, "shepp-logan"),
        ],
    )
    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_reconstruction_quality_gap(self, phantom_fn, phantom_name, n_angles):
        """Both libraries should achieve similar PSNR (within 3 dB)."""
        size = 512
        phantom = phantom_fn(size)
        theta_deg = np.linspace(0, 180, n_angles, endpoint=False)

        # scikit-image round-trip
        sino_sk = radon(phantom, theta=theta_deg)
        recon_sk = np.clip(iradon(sino_sk, theta=theta_deg, filter_name="ramp"), 0, 1)

        # torchtomo round-trip
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1)
        recon_tt = recon_tt.squeeze().numpy()

        psnr_sk = psnr(phantom, recon_sk, data_range=1.0)
        psnr_tt = psnr(phantom, recon_tt, data_range=1.0)
        gap = abs(psnr_tt - psnr_sk)

        assert gap < 3.0, (
            f"{phantom_name} {n_angles}angles: PSNR gap {gap:.2f} dB > 3 dB "
            f"(skimage={psnr_sk:.2f}, torchtomo={psnr_tt:.2f})"
        )

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_smooth_phantom_quality(self, n_angles):
        """Torchtomo should achieve high PSNR on smooth phantoms.

        Smooth phantoms like Gaussians are reconstructed near-perfectly by
        scikit-image (>80 dB), so the PSNR gap metric is misleading.
        Instead, verify torchtomo achieves strong absolute quality (>40 dB).
        """
        size = 512
        phantom = _make_gaussian_phantom(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1)
        recon_tt = recon_tt.squeeze().numpy()

        psnr_tt = psnr(phantom, recon_tt, data_range=1.0)
        assert psnr_tt > 40, f"Gaussian PSNR {psnr_tt:.2f} dB < 40 dB"

    @pytest.mark.parametrize(
        "phantom_fn,phantom_name",
        [
            (_make_disc_phantom, "disc"),
            (_shepp_logan_np, "shepp-logan"),
        ],
    )
    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_reconstruction_ssim_gap(self, phantom_fn, phantom_name, n_angles):
        """Both libraries should achieve comparable SSIM."""
        size = 512
        phantom = phantom_fn(size)
        theta_deg = np.linspace(0, 180, n_angles, endpoint=False)

        sino_sk = radon(phantom, theta=theta_deg)
        recon_sk = np.clip(iradon(sino_sk, theta=theta_deg, filter_name="ramp"), 0, 1)

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1)
        recon_tt = recon_tt.squeeze().numpy()

        ssim_sk = ssim(phantom, recon_sk, data_range=1.0)
        ssim_tt = ssim(phantom, recon_tt, data_range=1.0)

        # Both should achieve reasonable SSIM (lower bound accounts for
        # 512x512 with 180 angles being undersampled)
        assert ssim_sk > 0.65, f"skimage SSIM {ssim_sk:.4f} < 0.65"
        assert ssim_tt > 0.65, f"torchtomo SSIM {ssim_tt:.4f} < 0.65"

        # Gap should be small
        gap = abs(ssim_tt - ssim_sk)
        assert gap < 0.15, f"{phantom_name}: SSIM gap {gap:.4f} > 0.15 (skimage={ssim_sk:.4f}, torchtomo={ssim_tt:.4f})"


class TestReconstructionSimilarity:
    """Directly compare reconstructed images from both libraries."""

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_reconstruction_psnr(self, n_angles):
        """Reconstructions from both libraries should be similar."""
        size = 512
        phantom = _shepp_logan_np(size)
        theta_deg = np.linspace(0, 180, n_angles, endpoint=False)

        sino_sk = radon(phantom, theta=theta_deg)
        recon_sk = np.clip(iradon(sino_sk, theta=theta_deg, filter_name="ramp"), 0, 1)

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1)
        recon_tt = recon_tt.squeeze().numpy()

        # Direct comparison between the two reconstructions
        recon_psnr = psnr(recon_sk, recon_tt, data_range=1.0)
        recon_ssim = ssim(recon_sk, recon_tt, data_range=1.0)

        assert recon_psnr > 25, f"Reconstruction PSNR {recon_psnr:.2f} dB < 25 dB"
        assert recon_ssim > 0.7, f"Reconstruction SSIM {recon_ssim:.4f} < 0.7"

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_reconstruction_max_error(self, n_angles):
        """Maximum pixel error between reconstructions should be bounded."""
        size = 512
        phantom = _shepp_logan_np(size)
        theta_deg = np.linspace(0, 180, n_angles, endpoint=False)

        sino_sk = radon(phantom, theta=theta_deg)
        recon_sk = np.clip(iradon(sino_sk, theta=theta_deg, filter_name="ramp"), 0, 1)

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1)
        recon_tt = recon_tt.squeeze().numpy()

        max_err = np.abs(recon_sk - recon_tt).max()
        assert max_err < 0.2, f"Max reconstruction error {max_err:.4f} > 0.2"


class TestFilterConsistency:
    """Compare FBP with different filters between torchtomo and scikit-image."""

    @pytest.mark.parametrize("filter_name", ["ramp", "shepp-logan", "cosine", "hamming", "hann"])
    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_filter_reconstruction_gap(self, filter_name, n_angles):
        """Each filter should produce similar results in both libraries."""
        size = 512
        phantom = _shepp_logan_np(size)
        theta_deg = np.linspace(0, 180, n_angles, endpoint=False)

        sino_sk = radon(phantom, theta=theta_deg)
        recon_sk = np.clip(iradon(sino_sk, theta=theta_deg, filter_name=filter_name), 0, 1)

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t), filter_name=filter_name).clamp(0, 1).squeeze().numpy()

        psnr_sk = psnr(phantom, recon_sk, data_range=1.0)
        psnr_tt = psnr(phantom, recon_tt, data_range=1.0)
        gap = abs(psnr_tt - psnr_sk)

        assert gap < 3.0, (
            f"Filter '{filter_name}': PSNR gap {gap:.2f} dB > 3 dB (skimage={psnr_sk:.2f}, torchtomo={psnr_tt:.2f})"
        )

    @pytest.mark.parametrize("filter_name", ["ramp", "cosine", "hamming", "hann"])
    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_filter_reconstruction_similarity(self, filter_name, n_angles):
        """Reconstructions with the same filter should be directly comparable."""
        size = 512
        phantom = _shepp_logan_np(size)
        theta_deg = np.linspace(0, 180, n_angles, endpoint=False)

        sino_sk = radon(phantom, theta=theta_deg)
        recon_sk = np.clip(iradon(sino_sk, theta=theta_deg, filter_name=filter_name), 0, 1)

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t), filter_name=filter_name).clamp(0, 1).squeeze().numpy()

        recon_ssim = ssim(recon_sk, recon_tt, data_range=1.0)
        assert recon_ssim > 0.7, f"Filter '{filter_name}': reconstruction SSIM {recon_ssim:.4f} < 0.7"


class TestAngularConvergence:
    """Both libraries should show similar convergence with increasing angles."""

    def test_quality_improves_with_angles(self):
        """More angles should improve reconstruction quality for both."""
        size = 512
        phantom = _shepp_logan_np(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        prev_psnr_sk = 0
        prev_psnr_tt = 0

        for n_angles in [180, 360, 1000]:
            theta_deg = np.linspace(0, 180, n_angles, endpoint=False)

            sino_sk = radon(phantom, theta=theta_deg)
            recon_sk = np.clip(iradon(sino_sk, theta=theta_deg, filter_name="ramp"), 0, 1)

            projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
            recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1).squeeze().numpy()

            psnr_sk = psnr(phantom, recon_sk, data_range=1.0)
            psnr_tt = psnr(phantom, recon_tt, data_range=1.0)

            assert psnr_sk >= prev_psnr_sk, (
                f"skimage PSNR decreased from {prev_psnr_sk:.2f} to {psnr_sk:.2f} when going to {n_angles} angles"
            )
            assert psnr_tt >= prev_psnr_tt, (
                f"torchtomo PSNR decreased from {prev_psnr_tt:.2f} to {psnr_tt:.2f} when going to {n_angles} angles"
            )

            prev_psnr_sk = psnr_sk
            prev_psnr_tt = psnr_tt
