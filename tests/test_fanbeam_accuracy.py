"""Independent accuracy checks for fan-beam geometry and reconstruction."""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr

from torchtomo import FanBeam, ParallelBeam, shepp_logan


def _make_disc_phantom(size=128, radius=0.3):
    """Binary disc phantom in normalized [-1, 1] coordinates."""
    coords = np.linspace(-1, 1, size)
    y, x = np.meshgrid(coords, coords, indexing="ij")
    return ((x**2 + y**2) <= radius**2).astype(np.float32)


def _make_gaussian_phantom(size=128):
    """Smooth phantom masked near the inscribed circle."""
    coords = np.linspace(-1, 1, size)
    y, x = np.meshgrid(coords, coords, indexing="ij")
    phantom = np.exp(-(x**2 + y**2) / (2 * 0.2**2)).astype(np.float32)
    phantom[x**2 + y**2 > 0.95**2] = 0
    return phantom


def _analytic_disc_sinogram(projector: FanBeam, radius: float) -> torch.Tensor:
    """Analytic fan-beam sinogram for a centered uniform disc.

    For each ray, the line integral is the chord length through the disc:
    2 * sqrt(r^2 - d^2), where d is the perpendicular distance from the
    ray to the origin.
    """
    angles = projector.angles.cpu().numpy()
    scale = 2.0 / projector.img_size
    src_dist = projector.src_dist * scale
    det_dist = projector.det_dist * scale
    det_width = projector.det_width * scale
    det_offsets = np.linspace(
        -det_width / 2, det_width / 2, projector.n_det, dtype=np.float64
    )

    sinogram = np.zeros((projector.n_angles, projector.n_det), dtype=np.float32)

    for i, angle in enumerate(angles):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        src = np.array([-src_dist * sin_a, src_dist * cos_a], dtype=np.float64)
        det_center = np.array([det_dist * sin_a, -det_dist * cos_a], dtype=np.float64)
        det_dir = np.array([cos_a, sin_a], dtype=np.float64)
        det_points = det_center[None, :] + det_offsets[:, None] * det_dir[None, :]

        ray_dir = det_points - src[None, :]
        ray_norm = np.linalg.norm(ray_dir, axis=1)

        # Distance from the origin to the infinite line through source/detector.
        dist = np.abs(src[0] * det_points[:, 1] - src[1] * det_points[:, 0]) / ray_norm

        chord = np.zeros_like(dist)
        inside = dist < radius
        chord[inside] = 2.0 * np.sqrt(radius**2 - dist[inside] ** 2)
        sinogram[i] = chord.astype(np.float32)

    return torch.from_numpy(sinogram).unsqueeze(0).unsqueeze(0)


def _dice_score(pred: np.ndarray, truth: np.ndarray) -> float:
    intersection = np.logical_and(pred, truth).sum()
    return (2.0 * intersection) / (pred.sum() + truth.sum())


class TestFanBeamAccuracy:
    """Validate fan-beam paths against independent references."""

    def test_forward_matches_analytic_disc(self):
        size = 128
        radius = 0.3
        phantom = _make_disc_phantom(size=size, radius=radius)
        phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

        projector = FanBeam(
            img_size=size,
            n_angles=180,
            n_det=int(size * 1.5),
            src_dist=size * 2,
            det_dist=size * 2,
        )

        sino_tt = projector.forward(phantom_t).squeeze().numpy()
        sino_ref = _analytic_disc_sinogram(projector, radius=radius).squeeze().numpy()

        corr = np.corrcoef(sino_tt.ravel(), sino_ref.ravel())[0, 1]
        rmse = np.sqrt(np.mean((sino_tt - sino_ref) ** 2))
        peak = sino_ref.max()

        assert corr > 0.999, f"Fan-beam sinogram correlation {corr:.6f} < 0.999"
        assert rmse / peak < 0.01, (
            f"Fan-beam sinogram relative RMSE {(rmse / peak):.4f} >= 0.01"
        )

    def test_fbp_reconstructs_analytic_disc(self):
        size = 128
        radius = 0.3
        phantom = _make_disc_phantom(size=size, radius=radius)
        sino_ref = _analytic_disc_sinogram(
            FanBeam(
                img_size=size,
                n_angles=180,
                n_det=int(size * 1.5),
                src_dist=size * 2,
                det_dist=size * 2,
            ),
            radius=radius,
        )

        projector = FanBeam(
            img_size=size,
            n_angles=180,
            n_det=int(size * 1.5),
            src_dist=size * 2,
            det_dist=size * 2,
        )
        recon = projector.fbp(sino_ref).clamp(0, 1).squeeze().numpy()

        recon_psnr = psnr(phantom, recon, data_range=1.0)
        recon_mask = recon >= 0.5
        truth_mask = phantom >= 0.5
        dice = _dice_score(recon_mask, truth_mask)

        assert recon_psnr > 31.0, (
            f"Analytic fan-beam disc PSNR {recon_psnr:.2f} dB < 31"
        )
        assert dice > 0.995, f"Analytic fan-beam disc Dice {dice:.4f} < 0.995"

    def test_round_trip_quality_tracks_parallel_beam(self):
        size = 128
        phantoms = {
            "shepp-logan": shepp_logan(size).squeeze().numpy().astype(np.float32),
            "gaussian": _make_gaussian_phantom(size),
        }

        parallel = ParallelBeam(img_size=size, n_angles=180, n_det=size)
        fan = FanBeam(
            img_size=size,
            n_angles=360,
            n_det=int(size * 1.5),
            src_dist=size * 2,
            det_dist=size * 2,
        )

        for name, phantom in phantoms.items():
            phantom_t = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)

            recon_parallel = parallel.fbp(parallel.forward(phantom_t)).clamp(0, 1)
            recon_fan = fan.fbp(fan.forward(phantom_t)).clamp(0, 1)

            psnr_parallel = psnr(
                phantom, recon_parallel.squeeze().numpy(), data_range=1.0
            )
            psnr_fan = psnr(phantom, recon_fan.squeeze().numpy(), data_range=1.0)

            assert psnr_fan >= psnr_parallel - 3.0, (
                f"{name} fan-beam PSNR {psnr_fan:.2f} dB is more than 3 dB below "
                f"parallel-beam {psnr_parallel:.2f} dB"
            )
