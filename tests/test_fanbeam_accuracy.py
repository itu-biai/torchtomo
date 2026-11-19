"""Independent accuracy checks for fan-beam geometry and reconstruction."""

import numpy as np
import torch

from torchtomo import FanBeam, ParallelBeam, shepp_logan


def _psnr(truth, recon, data_range=1.0):
    mse = float(np.mean((truth - recon) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range**2 / mse)


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
    # One pixel is the pitch of the image lattice, linspace(-1, 1, img_size).
    scale = 2.0 / (projector.img_size - 1)
    src_dist = projector.src_dist * scale
    det_dist = projector.det_dist * scale
    det_width = projector.det_width * scale
    det_offsets = np.linspace(-det_width / 2, det_width / 2, projector.n_det, dtype=np.float64)

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
        assert rmse / peak < 0.01, f"Fan-beam sinogram relative RMSE {(rmse / peak):.4f} >= 0.01"

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
        raw = projector.fbp(sino_ref).squeeze().numpy()
        recon = np.clip(raw, 0, 1)

        recon_psnr = _psnr(phantom, recon, data_range=1.0)
        recon_mask = recon >= 0.5
        truth_mask = phantom >= 0.5
        dice = _dice_score(recon_mask, truth_mask)

        # PSNR and Dice against the binary disc hinge on a few edge pixels; the
        # value inside the disc and the radius from its mass measure the scale.
        coords = np.linspace(-1, 1, size)
        y, x = np.meshgrid(coords, coords, indexing="ij")
        centre = np.hypot(x, y)
        interior = float(raw[centre < radius - 0.08].mean())
        pixel_area = (2 / (size - 1)) ** 2
        mass_radius = float(np.sqrt(raw[centre < radius + 0.15].sum() * pixel_area / interior / np.pi))

        assert recon_psnr > 30.5, f"Analytic fan-beam disc PSNR {recon_psnr:.2f} dB < 30.5"
        assert dice > 0.99, f"Analytic fan-beam disc Dice {dice:.4f} < 0.99"
        assert abs(interior - 1.0) < 0.003, f"Analytic fan-beam disc value {interior:.4f}, expected 1"
        assert abs(mass_radius - radius) < 0.002, f"Analytic fan-beam disc radius {mass_radius:.4f}, expected {radius}"

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

            psnr_parallel = _psnr(phantom, recon_parallel.squeeze().numpy(), data_range=1.0)
            psnr_fan = _psnr(phantom, recon_fan.squeeze().numpy(), data_range=1.0)

            # The guard is that fan beam tracks parallel beam on a phantom with
            # real structure. On the smooth Gaussian the Ram-Lak ramp makes the
            # parallel FBP all but exact (72 dB against the fan's 64), and what is
            # left there is the fan's per-ray sampling, not an error a
            # reconstruction would show, so an absolute floor carries that case.
            floor = min(psnr_parallel - 3.0, 60.0)
            assert psnr_fan >= floor, (
                f"{name} fan-beam PSNR {psnr_fan:.2f} dB is below {floor:.2f} dB, "
                f"with parallel-beam at {psnr_parallel:.2f} dB"
            )

    def test_default_geometry_is_size_aware(self):
        projector = FanBeam(img_size=512, n_angles=8, n_samples=8)
        assert projector.src_dist == 1024
        assert projector.det_dist == 1024
        assert projector.n_det == 768
        mag = (projector.src_dist + projector.det_dist) / projector.src_dist
        assert abs(projector.det_width / mag - 1.5 * 512) < 1e-6

    def test_fbp_circle_psnr_within_one_db_of_parallel_at_512(self):
        """Noiseless Shepp-Logan, 360 views, PSNR over the inscribed circle."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        size, n_angles = 512, 360
        phantom = shepp_logan(size, device=device)
        parallel = ParallelBeam(img_size=size, n_angles=n_angles).to(device)
        fan = FanBeam(img_size=size, n_angles=n_angles).to(device)
        mask = parallel.circle_mask.bool()
        recon_p = parallel.fbp(parallel.forward(phantom))
        recon_f = fan.fbp(fan.forward(phantom))
        mse_p = ((recon_p - phantom) ** 2)[:, :, mask].mean().clamp_min(1e-12)
        mse_f = ((recon_f - phantom) ** 2)[:, :, mask].mean().clamp_min(1e-12)
        psnr_p = float(-10 * torch.log10(mse_p))
        psnr_f = float(-10 * torch.log10(mse_f))
        # The Ram-Lak ramp is worth 1.1 dB to parallel beam here and 0.1 dB to fan
        # beam, which was already close to it: a 1.5x wider padded detector leaves
        # far less of the projection mean in the DC bin. The fan's remaining 0.7 dB
        # is its per-ray sampling.
        assert psnr_f >= psnr_p - 1.0, f"fan {psnr_f:.2f} dB vs parallel {psnr_p:.2f} dB"


def test_lengths_in_lattice_pixels_give_physical_scale():
    """Distances in millimetres over a 1 mm lattice pixel reconstruct a disc at its true size and value.

    The sinogram is computed analytically in millimetres, independently of the projector's
    internal units: a disc of radius 30 mm and attenuation 1 per normalised length, seen by a
    flat detector of 0.75 mm bins. A pixel is the pitch of linspace(-1, 1, size), so the
    reconstruction's half extent is (size - 1) / 2 mm.
    """
    size, radius_mm, pitch_mm, n_det, n_angles = 96, 30.0, 0.75, 220, 720
    src_mm, det_mm = 180.0, 120.0
    half_extent_mm = (size - 1) / 2
    angles = np.arange(n_angles) * (2 * np.pi / n_angles)
    offsets = (np.arange(n_det) - (n_det - 1) / 2) * pitch_mm
    # A ray from the source to detector offset u passes the axis at distance src * u / hypot(src + det, u).
    distance = src_mm * np.abs(offsets) / np.hypot(src_mm + det_mm, offsets)
    chord = 2 * np.sqrt(np.clip(radius_mm**2 - distance**2, 0, None)) / half_extent_mm
    sinogram = torch.from_numpy(np.tile(chord, (n_angles, 1)).astype(np.float32))[None, None]

    projector = FanBeam(
        img_size=size,
        n_angles=n_angles,
        n_det=n_det,
        src_dist=src_mm,
        det_dist=det_mm,
        det_spacing=pitch_mm,
        angles=torch.from_numpy(angles),
        circle=False,
    )
    image = projector.fbp(sinogram, "ramp")[0, 0].numpy()

    coords = np.linspace(-half_extent_mm, half_extent_mm, size)
    y, x = np.meshgrid(coords, coords, indexing="ij")
    centre = np.hypot(x, y)
    interior = float(image[centre < radius_mm - 3].mean())
    radius_estimate = float(np.sqrt((image > interior / 2).sum() / np.pi))
    assert abs(interior - 1.0) < 0.01, f"disc value {interior:.4f}, expected 1"
    assert abs(radius_estimate - radius_mm) < 0.1, f"disc radius {radius_estimate:.3f} px, expected {radius_mm}"
