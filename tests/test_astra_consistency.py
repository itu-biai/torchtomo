"""Tests verifying torchtomo consistency with ASTRA Toolbox.

ASTRA Toolbox is a well-established, high-performance CT reconstruction
library. These tests compare torchtomo's forward projections and
reconstructions against ASTRA for both parallel and fan beam geometries.
"""

import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

astra = pytest.importorskip("astra")

from torchtomo import FanBeam, ParallelBeam, shepp_logan


# ---------------------------------------------------------------------------
# Phantoms
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ASTRA helpers
# ---------------------------------------------------------------------------


def _astra_parallel_sinogram(phantom, n_angles):
    """Compute parallel beam sinogram using ASTRA."""
    size = phantom.shape[0]
    theta = np.linspace(0, np.pi, n_angles, endpoint=False)
    proj_geom = astra.create_proj_geom("parallel", 1.0, size, theta)
    vol_geom = astra.create_vol_geom(size, size)
    proj_id = astra.create_projector("line", proj_geom, vol_geom)
    sino_id, sino = astra.create_sino(phantom, proj_id)
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)
    return sino


def _astra_parallel_fbp(phantom, n_angles, filter_type="Ram-Lak"):
    """Compute parallel beam FBP reconstruction using ASTRA."""
    size = phantom.shape[0]
    theta = np.linspace(0, np.pi, n_angles, endpoint=False)
    proj_geom = astra.create_proj_geom("parallel", 1.0, size, theta)
    vol_geom = astra.create_vol_geom(size, size)
    proj_id = astra.create_projector("line", proj_geom, vol_geom)
    sino_id, sino = astra.create_sino(phantom, proj_id)

    rec_id = astra.data2d.create("-vol", vol_geom)
    cfg = astra.astra_dict("FBP")
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectionDataId"] = sino_id
    cfg["ProjectorId"] = proj_id
    cfg["option"] = {"FilterType": filter_type}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recon = astra.data2d.get(rec_id)

    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)
    astra.projector.delete(proj_id)
    astra.algorithm.delete(alg_id)
    return recon


def _astra_fanbeam_sinogram(phantom, n_angles, n_det, src_dist, det_dist, det_width):
    """Compute fan beam sinogram using ASTRA."""
    size = phantom.shape[0]
    det_spacing = det_width / n_det
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    proj_geom = astra.create_proj_geom(
        "fanflat", det_spacing, n_det, angles, src_dist, det_dist
    )
    vol_geom = astra.create_vol_geom(size, size)
    proj_id = astra.create_projector("line_fanflat", proj_geom, vol_geom)
    sino_id, sino = astra.create_sino(phantom, proj_id)
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)
    return sino


# ---------------------------------------------------------------------------
# Parallel Beam: Sinogram Consistency
# ---------------------------------------------------------------------------


class TestParallelSinogramConsistency:
    """Compare parallel beam forward projections: torchtomo vs ASTRA."""

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_sinogram_correlation(self, n_angles):
        """Sinograms should be highly correlated after scaling."""
        size = 512
        phantom = _shepp_logan_np(size)

        sino_astra = _astra_parallel_sinogram(phantom, n_angles)

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        sino_tt = projector.forward(phantom_t).squeeze().numpy()
        sino_tt_scaled = sino_tt * (size / 2)

        corr = np.corrcoef(sino_astra.ravel(), sino_tt_scaled.ravel())[0, 1]
        assert corr > 0.99, f"Sinogram correlation {corr:.4f} < 0.99"

    @pytest.mark.parametrize("n_angles", [180, 360])
    def test_sinogram_peak_values(self, n_angles):
        """Peak projection values should match after scaling."""
        size = 512
        phantom = _make_disc_phantom(size)

        sino_astra = _astra_parallel_sinogram(phantom, n_angles)

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        sino_tt = projector.forward(phantom_t).squeeze().numpy()

        peak_astra = sino_astra.max()
        peak_tt = sino_tt.max() * (size / 2)
        rel_diff = abs(peak_astra - peak_tt) / peak_astra
        assert rel_diff < 0.05, f"Peak sinogram relative diff {rel_diff:.4f} > 0.05"


# ---------------------------------------------------------------------------
# Parallel Beam: FBP Consistency
# ---------------------------------------------------------------------------


class TestParallelFBPConsistency:
    """Compare parallel beam FBP reconstructions: torchtomo vs ASTRA."""

    @pytest.mark.parametrize(
        "phantom_fn,phantom_name",
        [
            (_make_disc_phantom, "disc"),
            (_shepp_logan_np, "shepp-logan"),
        ],
    )
    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_reconstruction_quality_gap(self, phantom_fn, phantom_name, n_angles):
        """Both libraries should achieve similar PSNR (within 5 dB).

        ASTRA uses a line-intersection projector model while torchtomo uses
        rotation-based grid sampling, so small quality differences are expected.
        """
        size = 512
        phantom = phantom_fn(size)

        recon_astra = np.clip(_astra_parallel_fbp(phantom, n_angles), 0, 1)

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1)
        recon_tt = recon_tt.squeeze().numpy()

        psnr_astra = psnr(phantom, recon_astra, data_range=1.0)
        psnr_tt = psnr(phantom, recon_tt, data_range=1.0)
        gap = abs(psnr_tt - psnr_astra)

        assert gap < 5.0, (
            f"{phantom_name} {n_angles}angles: PSNR gap {gap:.2f} dB > 5 dB "
            f"(astra={psnr_astra:.2f}, torchtomo={psnr_tt:.2f})"
        )

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_smooth_phantom_quality(self, n_angles):
        """Torchtomo should achieve high PSNR on smooth phantoms (>40 dB)."""
        size = 512
        phantom = _make_gaussian_phantom(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1)
        recon_tt = recon_tt.squeeze().numpy()

        psnr_tt = psnr(phantom, recon_tt, data_range=1.0)
        assert psnr_tt > 40, f"Gaussian PSNR {psnr_tt:.2f} dB < 40 dB"

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_torchtomo_ssim_quality(self, n_angles):
        """torchtomo should achieve reasonable SSIM on Shepp-Logan."""
        size = 512
        phantom = _shepp_logan_np(size)

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1)
        recon_tt = recon_tt.squeeze().numpy()

        ssim_tt = ssim(phantom, recon_tt, data_range=1.0)
        assert ssim_tt > 0.65, f"torchtomo SSIM {ssim_tt:.4f} < 0.65"


# ---------------------------------------------------------------------------
# Parallel Beam: Direct Reconstruction Comparison
# ---------------------------------------------------------------------------


class TestParallelReconstructionSimilarity:
    """Directly compare reconstructed images from both libraries."""

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_reconstruction_psnr(self, n_angles):
        """Reconstructions from both libraries should be similar in PSNR."""
        size = 512
        phantom = _shepp_logan_np(size)

        recon_astra = np.clip(_astra_parallel_fbp(phantom, n_angles), 0, 1)

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1)
        recon_tt = recon_tt.squeeze().numpy()

        recon_psnr = psnr(recon_astra, recon_tt, data_range=1.0)
        assert recon_psnr > 20, f"Reconstruction PSNR {recon_psnr:.2f} dB < 20 dB"

    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_reconstruction_max_error(self, n_angles):
        """Maximum pixel error between reconstructions should be bounded."""
        size = 512
        phantom = _shepp_logan_np(size)

        recon_astra = np.clip(_astra_parallel_fbp(phantom, n_angles), 0, 1)

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1)
        recon_tt = recon_tt.squeeze().numpy()

        max_err = np.abs(recon_astra - recon_tt).max()
        assert max_err < 0.4, f"Max reconstruction error {max_err:.4f} > 0.4"


# ---------------------------------------------------------------------------
# Parallel Beam: Filter Consistency
# ---------------------------------------------------------------------------


ASTRA_FILTER_MAP = {
    "ramp": "Ram-Lak",
    "shepp-logan": "Shepp-Logan",
    "cosine": "Cosine",
    "hamming": "Hamming",
    "hann": "Hann",
}


class TestParallelFilterConsistency:
    """Compare FBP with different filters: torchtomo vs ASTRA."""

    @pytest.mark.parametrize(
        "filter_name", ["ramp", "shepp-logan", "cosine", "hamming", "hann"]
    )
    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_filter_reconstruction_gap(self, filter_name, n_angles):
        """Each filter should produce similar results in both libraries."""
        size = 512
        phantom = _shepp_logan_np(size)

        astra_filter = ASTRA_FILTER_MAP[filter_name]
        recon_astra = np.clip(
            _astra_parallel_fbp(phantom, n_angles, filter_type=astra_filter), 0, 1
        )

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = (
            projector.fbp(projector.forward(phantom_t), filter_name=filter_name)
            .clamp(0, 1)
            .squeeze()
            .numpy()
        )

        psnr_astra = psnr(phantom, recon_astra, data_range=1.0)
        psnr_tt = psnr(phantom, recon_tt, data_range=1.0)
        gap = abs(psnr_tt - psnr_astra)

        assert gap < 5.0, (
            f"Filter '{filter_name}': PSNR gap {gap:.2f} dB > 5 dB "
            f"(astra={psnr_astra:.2f}, torchtomo={psnr_tt:.2f})"
        )

    @pytest.mark.parametrize("filter_name", ["ramp", "cosine", "hamming", "hann"])
    @pytest.mark.parametrize("n_angles", [180, 360, 1000])
    def test_filter_reconstruction_similarity(self, filter_name, n_angles):
        """Reconstructions with the same filter should be directly comparable."""
        size = 512
        phantom = _shepp_logan_np(size)

        astra_filter = ASTRA_FILTER_MAP[filter_name]
        recon_astra = np.clip(
            _astra_parallel_fbp(phantom, n_angles, filter_type=astra_filter), 0, 1
        )

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
        recon_tt = (
            projector.fbp(projector.forward(phantom_t), filter_name=filter_name)
            .clamp(0, 1)
            .squeeze()
            .numpy()
        )

        recon_psnr = psnr(recon_astra, recon_tt, data_range=1.0)
        assert recon_psnr > 20, (
            f"Filter '{filter_name}': reconstruction PSNR {recon_psnr:.2f} dB < 20 dB"
        )


# ---------------------------------------------------------------------------
# Parallel Beam: Angular Convergence
# ---------------------------------------------------------------------------


class TestParallelAngularConvergence:
    """Both libraries should show similar convergence with increasing angles."""

    def test_quality_improves_with_angles(self):
        """More angles should improve reconstruction quality for both."""
        size = 512
        phantom = _shepp_logan_np(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        prev_psnr_astra = 0
        prev_psnr_tt = 0

        for n_angles in [180, 360, 1000]:
            recon_astra = np.clip(_astra_parallel_fbp(phantom, n_angles), 0, 1)

            projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size)
            recon_tt = (
                projector.fbp(projector.forward(phantom_t))
                .clamp(0, 1)
                .squeeze()
                .numpy()
            )

            psnr_astra = psnr(phantom, recon_astra, data_range=1.0)
            psnr_tt = psnr(phantom, recon_tt, data_range=1.0)

            assert psnr_astra >= prev_psnr_astra, (
                f"ASTRA PSNR decreased from {prev_psnr_astra:.2f} to "
                f"{psnr_astra:.2f} when going to {n_angles} angles"
            )
            assert psnr_tt >= prev_psnr_tt, (
                f"torchtomo PSNR decreased from {prev_psnr_tt:.2f} to "
                f"{psnr_tt:.2f} when going to {n_angles} angles"
            )

            prev_psnr_astra = psnr_astra
            prev_psnr_tt = psnr_tt


# ---------------------------------------------------------------------------
# Fan Beam: Sinogram Consistency
# ---------------------------------------------------------------------------


def _fanbeam_det_width(img_size, src_dist, det_dist):
    """Compute torchtomo's default fan beam detector width."""
    magnification = (src_dist + det_dist) / src_dist
    return 1.5 * magnification * img_size


class TestFanBeamSinogramConsistency:
    """Compare fan beam forward projections: torchtomo vs ASTRA."""

    @pytest.mark.parametrize("n_angles", [360, 720])
    def test_sinogram_correlation(self, n_angles):
        """Fan beam sinograms should be highly correlated.

        ASTRA and torchtomo use opposite angle rotation directions for fan beam,
        so we compare with the angle axis reversed.
        """
        size = 256
        n_det = 400
        src_dist = 500.0
        det_dist = 500.0
        phantom = _shepp_logan_np(size)

        det_width = _fanbeam_det_width(size, src_dist, det_dist)
        sino_astra = _astra_fanbeam_sinogram(
            phantom, n_angles, n_det, src_dist, det_dist, det_width
        )

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = FanBeam(
            img_size=size,
            n_angles=n_angles,
            n_det=n_det,
            src_dist=src_dist,
            det_dist=det_dist,
        )
        sino_tt = projector.forward(phantom_t).squeeze().numpy()
        sino_tt_scaled = sino_tt * (size / 2)

        # Account for opposite angle rotation convention
        sino_tt_reversed = sino_tt_scaled[::-1]

        corr = np.corrcoef(sino_astra.ravel(), sino_tt_reversed.ravel())[0, 1]
        assert corr > 0.99, f"Fan beam sinogram correlation {corr:.4f} < 0.99"

    @pytest.mark.parametrize("n_angles", [360, 720])
    def test_sinogram_peak_values(self, n_angles):
        """Peak projection values should match after scaling."""
        size = 256
        n_det = 400
        src_dist = 500.0
        det_dist = 500.0
        phantom = _make_disc_phantom(size)

        det_width = _fanbeam_det_width(size, src_dist, det_dist)
        sino_astra = _astra_fanbeam_sinogram(
            phantom, n_angles, n_det, src_dist, det_dist, det_width
        )

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = FanBeam(
            img_size=size,
            n_angles=n_angles,
            n_det=n_det,
            src_dist=src_dist,
            det_dist=det_dist,
        )
        sino_tt = projector.forward(phantom_t).squeeze().numpy()

        peak_astra = sino_astra.max()
        peak_tt = sino_tt.max() * (size / 2)
        rel_diff = abs(peak_astra - peak_tt) / peak_astra
        assert rel_diff < 0.05, (
            f"Fan beam peak sinogram relative diff {rel_diff:.4f} > 0.05"
        )


# ---------------------------------------------------------------------------
# Fan Beam: FBP Self-Consistency
# ---------------------------------------------------------------------------


class TestFanBeamFBPQuality:
    """Verify fan beam FBP reconstruction quality.

    ASTRA's CPU FBP doesn't support fan beam geometry, so we verify
    torchtomo's fan beam FBP via self-consistency (round-trip quality)
    and convergence to parallel beam at large source distances.
    """

    @pytest.mark.parametrize(
        "phantom_fn,phantom_name",
        [
            (_make_disc_phantom, "disc"),
            (_shepp_logan_np, "shepp-logan"),
        ],
    )
    @pytest.mark.parametrize("n_angles", [360, 720])
    def test_fbp_round_trip_quality(self, phantom_fn, phantom_name, n_angles):
        """Fan beam forward -> FBP should reconstruct with high PSNR."""
        size = 256
        phantom = phantom_fn(size)

        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
        projector = FanBeam(
            img_size=size,
            n_angles=n_angles,
            n_det=400,
            src_dist=500.0,
            det_dist=500.0,
        )
        recon_tt = projector.fbp(projector.forward(phantom_t)).clamp(0, 1)
        recon_tt = recon_tt.squeeze().numpy()

        psnr_tt = psnr(phantom, recon_tt, data_range=1.0)
        ssim_tt = ssim(phantom, recon_tt, data_range=1.0)

        assert psnr_tt > 25, (
            f"{phantom_name} {n_angles}angles: PSNR {psnr_tt:.2f} dB < 25 dB"
        )
        assert ssim_tt > 0.85, (
            f"{phantom_name} {n_angles}angles: SSIM {ssim_tt:.4f} < 0.85"
        )

    def test_quality_improves_with_angles(self):
        """More angles should improve fan beam reconstruction quality."""
        size = 256
        phantom = _shepp_logan_np(size)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        prev_psnr = 0
        for n_angles in [360, 720]:
            projector = FanBeam(
                img_size=size,
                n_angles=n_angles,
                n_det=400,
                src_dist=500.0,
                det_dist=500.0,
            )
            recon = projector.fbp(projector.forward(phantom_t)).clamp(0, 1)
            cur_psnr = psnr(phantom, recon.squeeze().numpy(), data_range=1.0)
            assert cur_psnr >= prev_psnr, (
                f"Fan beam PSNR decreased from {prev_psnr:.2f} to "
                f"{cur_psnr:.2f} at {n_angles} angles"
            )
            prev_psnr = cur_psnr


# ---------------------------------------------------------------------------
# Fan Beam: Cross-Library Forward Projection
# ---------------------------------------------------------------------------


class TestFanBeamCrossLibrary:
    """Use ASTRA's forward projection with torchtomo's FBP and vice versa."""

    def test_astra_sinogram_torchtomo_fbp(self):
        """torchtomo FBP should reconstruct from ASTRA's sinogram."""
        size = 256
        n_angles = 360
        n_det = 400
        src_dist = 500.0
        det_dist = 500.0
        phantom = _shepp_logan_np(size)

        det_width = _fanbeam_det_width(size, src_dist, det_dist)
        sino_astra = _astra_fanbeam_sinogram(
            phantom, n_angles, n_det, src_dist, det_dist, det_width
        )

        # Reverse angle order and scale to match torchtomo conventions
        sino_reversed = sino_astra[::-1].copy() / (size / 2)
        sino_t = torch.from_numpy(sino_reversed).unsqueeze(0).unsqueeze(0)

        projector = FanBeam(
            img_size=size,
            n_angles=n_angles,
            n_det=n_det,
            src_dist=src_dist,
            det_dist=det_dist,
        )
        recon = projector.fbp(sino_t).clamp(0, 1).squeeze().numpy()

        psnr_val = psnr(phantom, recon, data_range=1.0)
        assert psnr_val > 20, f"Cross-library FBP PSNR {psnr_val:.2f} dB < 20 dB"
