"""Tests verifying torchtomo consistency with LEAP (CUDA required).

LEAP (LivermorE AI Projector, LLNL) is an independent CUDA implementation, so it
checks torchtomo's parallel-beam geometry including its absolute scale: with the
image on [-1, 1] and a matching detector, the two forward projectors agree without
any fitted factor. LEAP turns the gantry the other way round, which is the one
convention that has to be negated.

    PYTHONPATH=src pytest benchmark/test_leap_consistency.py
"""

import numpy as np
import pytest
import torch

from torchtomo import FanBeam, ParallelBeam, shepp_logan

try:
    from leapctype import tomographicModels
except ImportError:
    tomographicModels = None

requires_leap = pytest.mark.skipif(
    tomographicModels is None or not torch.cuda.is_available(),
    reason="LEAP and CUDA required",
)


def _leap_parallel_beam(size, angles, slices=1, gpu=0):
    """A LEAP geometry in torchtomo's units: image and detector both on [-1, 1]."""
    pixel = 2.0 / size
    lct = tomographicModels()
    lct.set_gpu(gpu)
    lct.print_warnings = False
    phis = np.ascontiguousarray(-np.degrees(angles.detach().cpu().numpy()), dtype=np.float32)
    lct.set_parallelbeam(len(phis), slices, size, pixel, pixel, (slices - 1) / 2.0, (size - 1) / 2.0, phis)
    lct.set_volume(size, size, slices, pixel, pixel)
    return lct


def _relative_l2(reference, other):
    return float((reference - other).norm() / reference.norm())


def _psnr(truth, image, data_range=1.0):
    mse = float(((truth - image) ** 2).mean())
    return 10.0 * np.log10(data_range**2 / mse)


def _phantom(size):
    projector = ParallelBeam(img_size=size, n_angles=90).to("cuda")
    return shepp_logan(size).reshape(1, 1, size, size).cuda() * projector.circle_mask


@requires_leap
@pytest.mark.parametrize("n_angles", [90, 180, 360])
def test_forward_matches_leap(n_angles):
    """The sinograms agree outright, with no scale factor between them."""
    size = 512
    projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size).to("cuda")
    truth = _phantom(size)
    sinogram = projector.forward(truth)

    lct = _leap_parallel_beam(size, projector.angles)
    measured = torch.zeros(n_angles, 1, size, device="cuda")
    lct.project(measured, truth[:, 0].contiguous())
    measured = measured.permute(1, 0, 2).unsqueeze(1)

    assert _relative_l2(sinogram, measured) < 0.01


@requires_leap
@pytest.mark.parametrize("n_angles", [90, 180, 360])
def test_adjoint_matches_leap(n_angles):
    """torchtomo's matched adjoint agrees with LEAP's backprojection, scale included."""
    size = 512
    projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size).to("cuda")
    sinogram = projector.forward(_phantom(size))

    lct = _leap_parallel_beam(size, projector.angles)
    volume = torch.zeros(1, size, size, device="cuda")
    lct.backproject(sinogram[:, 0].permute(1, 0, 2).contiguous(), volume)

    adjoint = projector.adjoint(sinogram)
    assert _relative_l2(adjoint, volume.unsqueeze(1)) < 0.02
    scale = float(volume.abs().mean() / adjoint.abs().mean())
    assert 0.99 < scale < 1.01, f"backprojection scale {scale:.4f} is not unity"


@requires_leap
@pytest.mark.parametrize("n_angles", [90, 180, 360])
def test_fbp_quality_matches_leap(n_angles):
    """Both libraries reconstruct the same phantom to within a decibel."""
    size = 512
    projector = ParallelBeam(img_size=size, n_angles=n_angles, n_det=size).to("cuda")
    truth = _phantom(size)
    sinogram = projector.forward(truth)

    lct = _leap_parallel_beam(size, projector.angles)
    volume = torch.zeros(1, size, size, device="cuda")
    lct.FBP(sinogram[:, 0].permute(1, 0, 2).contiguous(), volume)

    mask = projector.circle_mask
    ours = _psnr(truth, projector.fbp(sinogram).clamp(0, 1) * mask)
    theirs = _psnr(truth, volume.unsqueeze(1).clamp(0, 1) * mask)
    assert abs(ours - theirs) < 1.0, (
        f"FBP PSNR gap {abs(ours - theirs):.2f} dB (torchtomo={ours:.2f}, leap={theirs:.2f})"
    )


@requires_leap
def test_fan_adapter_forward_matches_torchtomo():
    """LeapFanBeam is a drop-in: same image, same sinogram, no fitted scale."""
    try:
        from leap_projector import LeapFanBeam
    except ImportError:
        pytest.skip("leap_projector is not importable; put benchmark/ on PYTHONPATH")

    size = 256
    ours = FanBeam(img_size=size, n_angles=90).to("cuda")
    leap = LeapFanBeam(img_size=size, n_angles=90, angles=ours.angles.cpu()).to("cuda")
    truth = shepp_logan(size).reshape(1, 1, size, size).cuda() * ours.circle_mask
    assert _relative_l2(ours.forward(truth), leap.forward(truth)) < 0.02
