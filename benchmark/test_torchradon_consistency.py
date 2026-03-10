"""Tests verifying torchtomo consistency with torch-radon (CUDA required)."""

import numpy as np
import pytest
import torch

from torchtomo import FanBeam, ParallelBeam, shepp_logan

try:
    from torch_radon import Radon as TorchRadon
    from torch_radon import RadonFanbeam as TorchRadonFanbeam
except ImportError:
    TorchRadon = None
    TorchRadonFanbeam = None

requires_torchradon = pytest.mark.skipif(
    TorchRadon is None or not torch.cuda.is_available(),
    reason="torch-radon and CUDA required",
)


def _psnr(truth, recon, data_range=1.0):
    mse = float(np.mean((truth - recon) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range**2 / mse)


def _ssim(img1, img2, data_range=1.0):
    """Simplified mean-SSIM over 7x7 sliding window."""
    from scipy.ndimage import uniform_filter

    k1, k2, win = 0.01, 0.03, 7
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu1 = uniform_filter(img1.astype(np.float64), size=win)
    mu2 = uniform_filter(img2.astype(np.float64), size=win)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1_sq = uniform_filter(img1.astype(np.float64) ** 2, size=win) - mu1_sq
    s2_sq = uniform_filter(img2.astype(np.float64) ** 2, size=win) - mu2_sq
    s12 = uniform_filter(img1.astype(np.float64) * img2.astype(np.float64), size=win) - mu12
    num = (2 * mu12 + c1) * (2 * s12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (s1_sq + s2_sq + c2)
    return float(np.mean(num / den))


def _make_disc_phantom(size=256):
    x = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, x)
    phantom = np.zeros((size, size), dtype=np.float32)
    phantom[X**2 + Y**2 < 0.3**2] = 1.0
    return phantom


def _shepp_logan_np(size=256):
    return shepp_logan(size).squeeze().numpy()


# ---------------------------------------------------------------------------
# Parallel beam
# ---------------------------------------------------------------------------


@requires_torchradon
class TestParallelSinogramConsistency:
    """Compare parallel-beam forward projections."""

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_sinogram_correlation(self, n_angles):
        size = 512
        phantom = _make_disc_phantom(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        # torchtomo (CPU then compare)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        sino_tt = projector.forward(phantom_t).squeeze().numpy()

        # torch-radon (CUDA)
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        tr = TorchRadon(size, angles)
        sino_tr = tr.forward(phantom_t.cuda()).cpu().squeeze().numpy()

        corr = np.corrcoef(sino_tt.ravel(), sino_tr.ravel())[0, 1]
        assert corr > 0.99, f"Sinogram correlation {corr:.4f} < 0.99"

    def test_sinogram_peak_values(self):
        size = 512
        phantom = _make_disc_phantom(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        projector = ParallelBeam(img_size=size, n_angles=180, n_det=size)
        sino_tt = projector.forward(phantom_t).squeeze().numpy()

        angles = np.linspace(0, np.pi, 180, endpoint=False)
        tr = TorchRadon(size, angles)
        sino_tr = tr.forward(phantom_t.cuda()).cpu().squeeze().numpy()

        # Peaks should be proportional (different scaling conventions)
        ratio = sino_tr.max() / sino_tt.max()
        # Ratio should be consistent across all angles
        ratios_per_angle = sino_tr.max(axis=1) / (sino_tt.max(axis=1) + 1e-8)
        std_ratio = ratios_per_angle[sino_tt.max(axis=1) > 0.01].std()
        assert std_ratio < 0.1 * ratio, f"Scaling ratio inconsistent across angles: std={std_ratio:.4f}"


@requires_torchradon
class TestParallelBackprojectionConsistency:
    """Compare parallel-beam backprojections from the same sinogram."""

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_backprojection_correlation(self, n_angles):
        """Backprojections from same sinogram should be highly correlated."""
        size = 512
        phantom = _make_disc_phantom(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0).cuda()

        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        tr = TorchRadon(size, angles)
        sino_tr = tr.forward(phantom_t)

        # torch-radon backprojection
        bp_tr = tr.backprojection(sino_tr).cpu().squeeze().numpy()

        # torchtomo backprojection of same sinogram
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        bp_tt = projector.backward(sino_tr.cpu()).squeeze().numpy()

        corr = np.corrcoef(bp_tr.ravel(), bp_tt.ravel())[0, 1]
        assert corr > 0.95, f"Backprojection correlation {corr:.4f} < 0.95"


@requires_torchradon
class TestParallelRoundTripQuality:
    """Compare round-trip reconstruction quality against ground truth."""

    @pytest.mark.parametrize(
        "phantom_fn,phantom_name",
        [
            (_make_disc_phantom, "disc"),
            (_shepp_logan_np, "shepp-logan"),
        ],
    )
    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_torchtomo_fbp_on_torchradon_sinogram(self, phantom_fn, phantom_name, n_angles):
        """Torchtomo FBP on scale-normalised torch-radon sinograms."""
        size = 512
        phantom = phantom_fn(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        # torchtomo sinogram (reference scale)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        sino_tt = projector.forward(phantom_t)

        # torch-radon sinogram
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        tr = TorchRadon(size, angles)
        sino_tr = tr.forward(phantom_t.cuda()).cpu()

        # Scale-normalise: match torch-radon sinogram to torchtomo's scale
        scale = sino_tt.abs().mean() / (sino_tr.abs().mean() + 1e-8)
        sino_tr_scaled = sino_tr * scale

        # Reconstruct both with torchtomo FBP
        recon_tt = projector.fbp(sino_tt).clamp(0, 1).squeeze().numpy()
        recon_tr = projector.fbp(sino_tr_scaled).clamp(0, 1).squeeze().numpy()

        psnr_tt = _psnr(phantom, recon_tt)
        psnr_tr = _psnr(phantom, recon_tr)
        gap = abs(psnr_tt - psnr_tr)

        assert gap < 5.0, (
            f"{phantom_name} {n_angles}angles: PSNR gap {gap:.2f} dB > 5 dB "
            f"(torchtomo={psnr_tt:.2f}, torch-radon={psnr_tr:.2f})"
        )

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_cross_library_reconstruction_similarity(self, n_angles):
        """Reconstructions from both sinograms should look similar."""
        size = 512
        phantom = _shepp_logan_np(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        # torchtomo full pipeline
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1).squeeze().numpy()

        # torch-radon forward → scale-normalise → torchtomo FBP
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        tr = TorchRadon(size, angles)
        sino_tr = tr.forward(phantom_t.cuda()).cpu()
        sino_tt = projector.forward(phantom_t)
        scale = sino_tt.abs().mean() / (sino_tr.abs().mean() + 1e-8)
        recon_tr = projector.fbp(sino_tr * scale).clamp(0, 1).squeeze().numpy()

        recon_ssim = _ssim(recon_tt, recon_tr)
        assert recon_ssim > 0.7, f"Cross-library reconstruction SSIM {recon_ssim:.4f} < 0.7"


@requires_torchradon
class TestParallelReconstructionSimilarity:
    """Directly compare reconstructed images from both libraries."""

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_reconstruction_max_error(self, n_angles):
        """Maximum pixel error between scale-matched reconstructions."""
        size = 512
        phantom = _shepp_logan_np(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        sino_tt = projector.forward(phantom_t)
        recon_tt = projector.fbp(sino_tt).clamp(0, 1).squeeze().numpy()

        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        tr = TorchRadon(size, angles)
        sino_tr = tr.forward(phantom_t.cuda()).cpu()
        scale = sino_tt.abs().mean() / (sino_tr.abs().mean() + 1e-8)
        recon_tr = projector.fbp(sino_tr * scale).clamp(0, 1).squeeze().numpy()

        max_err = np.abs(recon_tt - recon_tr).max()
        # Higher tolerance than skimage comparison due to different interpolation
        # kernels between torch-radon (CUDA textures) and torchtomo (grid_sample)
        assert max_err < 0.75, f"Max reconstruction error {max_err:.4f} > 0.75"

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_reconstruction_ssim(self, n_angles):
        """SSIM between scale-matched reconstructions."""
        size = 512
        phantom = _shepp_logan_np(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        sino_tt = projector.forward(phantom_t)
        recon_tt = projector.fbp(sino_tt).clamp(0, 1).squeeze().numpy()

        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        tr = TorchRadon(size, angles)
        sino_tr = tr.forward(phantom_t.cuda()).cpu()
        scale = sino_tt.abs().mean() / (sino_tr.abs().mean() + 1e-8)
        recon_tr = projector.fbp(sino_tr * scale).clamp(0, 1).squeeze().numpy()

        ssim_val = _ssim(recon_tt, recon_tr)
        assert ssim_val > 0.7, f"Reconstruction SSIM {ssim_val:.4f} < 0.7"


@requires_torchradon
class TestParallelAngularConvergence:
    """Both libraries should show similar convergence with increasing angles."""

    def test_quality_improves_with_angles(self):
        size = 512
        phantom = _shepp_logan_np(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        prev_psnr_tt = 0
        prev_psnr_tr = 0

        for n_angles in [180, 360, 1000]:
            # torchtomo
            projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
            recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1).squeeze().numpy()

            # torch-radon (scale-normalised → torchtomo FBP)
            angles = np.linspace(0, np.pi, n_angles, endpoint=False)
            tr = TorchRadon(size, angles)
            sino_tr = tr.forward(phantom_t.cuda()).cpu()
            sino_tt = projector.forward(phantom_t)
            scale = sino_tt.abs().mean() / (sino_tr.abs().mean() + 1e-8)
            recon_tr = projector.fbp(sino_tr * scale).clamp(0, 1).squeeze().numpy()

            psnr_tt = _psnr(phantom, recon_tt)
            psnr_tr = _psnr(phantom, recon_tr)

            assert psnr_tt >= prev_psnr_tt, (
                f"torchtomo PSNR decreased from {prev_psnr_tt:.2f} to {psnr_tt:.2f} at {n_angles} angles"
            )
            assert psnr_tr >= prev_psnr_tr, (
                f"torch-radon PSNR decreased from {prev_psnr_tr:.2f} to {psnr_tr:.2f} at {n_angles} angles"
            )

            prev_psnr_tt = psnr_tt
            prev_psnr_tr = psnr_tr


# ---------------------------------------------------------------------------
# Fan beam
# ---------------------------------------------------------------------------


@requires_torchradon
class TestFanBeamSinogramConsistency:
    """Compare fan-beam forward projections."""

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_sinogram_correlation(self, n_angles):
        size = 512
        n_det = int(size * 1.5)
        src_dist = size * 2
        det_dist = size * 2
        phantom = _make_disc_phantom(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        # torchtomo
        proj_tt = FanBeam(
            img_size=size,
            n_angles=n_angles,
            n_det=n_det,
            src_dist=src_dist,
            det_dist=det_dist,
        )
        sino_tt = proj_tt.forward(phantom_t).squeeze().numpy()

        # torch-radon
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        tr = TorchRadonFanbeam(
            size,
            angles,
            source_distance=src_dist,
            det_distance=det_dist,
            det_count=n_det,
        )
        sino_tr = tr.forward(phantom_t.cuda()).cpu().squeeze().numpy()

        corr = np.corrcoef(sino_tt.ravel(), sino_tr.ravel())[0, 1]
        assert corr > 0.95, f"Fan-beam sinogram correlation {corr:.4f} < 0.95"


@requires_torchradon
class TestFanBeamRoundTripQuality:
    """Compare fan-beam round-trip reconstruction quality."""

    @pytest.mark.parametrize(
        "phantom_fn,phantom_name",
        [
            (_make_disc_phantom, "disc"),
            (_shepp_logan_np, "shepp-logan"),
        ],
    )
    def test_reconstruction_quality(self, phantom_fn, phantom_name):
        size = 512
        n_angles = 360
        n_det = int(size * 1.5)
        src_dist = size * 2
        det_dist = size * 2
        phantom = phantom_fn(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        # torchtomo round-trip
        proj_tt = FanBeam(
            img_size=size,
            n_angles=n_angles,
            n_det=n_det,
            src_dist=src_dist,
            det_dist=det_dist,
        )
        recon_tt = proj_tt.fbp(proj_tt.forward(phantom_t)).clamp(0, 1).squeeze().numpy()

        # torch-radon round-trip (forward + backprojection, no filtering available)
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        tr = TorchRadonFanbeam(
            size,
            angles,
            source_distance=src_dist,
            det_distance=det_dist,
            det_count=n_det,
        )
        sino_tr = tr.forward(phantom_t.cuda())
        recon_tr = tr.backprojection(sino_tr).cpu().squeeze().numpy()
        recon_tr = np.clip(recon_tr / (recon_tr.max() + 1e-8) * phantom.max(), 0, 1)

        psnr_tt = _psnr(phantom, recon_tt)
        psnr_tr = _psnr(phantom, recon_tr)

        # torchtomo with FBP should beat unfiltered backprojection
        assert psnr_tt > psnr_tr - 3.0, (
            f"{phantom_name}: torchtomo FBP {psnr_tt:.2f} dB should not be "
            f"more than 3 dB below torch-radon BP {psnr_tr:.2f} dB"
        )

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_backprojection_correlation(self, n_angles):
        """Backprojections from the same fan-beam sinogram should correlate."""
        size = 512
        n_det = int(size * 1.5)
        src_dist = size * 2
        det_dist = size * 2
        phantom = _make_disc_phantom(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0).cuda()

        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        tr = TorchRadonFanbeam(
            size,
            angles,
            source_distance=src_dist,
            det_distance=det_dist,
            det_count=n_det,
        )
        sino_tr = tr.forward(phantom_t)

        bp_tr = tr.backprojection(sino_tr).cpu().squeeze().numpy()

        proj_tt = FanBeam(
            img_size=size,
            n_angles=n_angles,
            n_det=n_det,
            src_dist=src_dist,
            det_dist=det_dist,
        )
        bp_tt = proj_tt.backward(sino_tr.cpu()).squeeze().numpy()

        corr = np.corrcoef(bp_tr.ravel(), bp_tt.ravel())[0, 1]
        assert corr > 0.90, f"Fan-beam backprojection correlation {corr:.4f} < 0.90"
