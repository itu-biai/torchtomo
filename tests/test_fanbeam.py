"""Tests for fan beam projector."""

import pytest
import torch

from torchtomo import FanBeam, shepp_logan


class TestFanBeam:
    def test_forward_shape(self, fan_projector, phantom_small):
        sinogram = fan_projector.forward(phantom_small)
        B, C, n_angles, n_det = sinogram.shape
        assert B == 1
        assert C == 1
        assert n_angles == fan_projector.n_angles
        assert n_det == fan_projector.n_det

    def test_backward_shape(self, fan_projector, phantom_small):
        sinogram = fan_projector.forward(phantom_small)
        recon = fan_projector.backward(sinogram)
        assert recon.shape == phantom_small.shape

    def test_fbp_shape(self, fan_projector, phantom_small):
        sinogram = fan_projector.forward(phantom_small)
        recon = fan_projector.fbp(sinogram)
        assert recon.shape == phantom_small.shape

    def test_fbp_quality(self):
        projector = FanBeam(
            img_size=256,
            n_angles=360,
            n_det=400,
            src_dist=500,
            det_dist=500,
        )
        phantom = shepp_logan(256)
        sinogram = projector.forward(phantom)
        recon = projector.fbp(sinogram)
        mse = ((recon - phantom) ** 2).mean().item()
        assert mse < 0.01

    def test_differentiable(self, fan_projector, phantom_small):
        phantom = phantom_small.clone().requires_grad_(True)
        sinogram = fan_projector.forward(phantom)
        loss = sinogram.sum()
        loss.backward()
        assert phantom.grad is not None
        assert phantom.grad.shape == phantom.shape

    def test_det_spacing_param(self, img_size):
        projector = FanBeam(
            img_size=img_size,
            n_angles=180,
            n_det=200,
            src_dist=500,
            det_dist=500,
            det_spacing=1.0,
        )
        assert projector.det_width == 199.0

    def test_det_width_param(self, img_size):
        projector = FanBeam(
            img_size=img_size,
            n_angles=180,
            n_det=200,
            src_dist=500,
            det_dist=500,
            det_width=300.0,
        )
        assert projector.det_width == 300.0

    def test_repr(self, fan_projector):
        repr_str = repr(fan_projector)
        assert "FanBeam" in repr_str
        assert "src_dist" in repr_str


def _backends():
    from torchtomo._nvrtc import runtime_available

    backends = [("torch", "cpu")]
    if torch.cuda.is_available() and runtime_available():
        backends.append(("cuda", "cuda"))
    return backends


@pytest.mark.parametrize("backend,device", _backends())
def test_corners_are_projected_without_the_circle(backend, device):
    """circle=False models the whole square: a corner pixel reaches the sinogram and the adjoint reaches it."""
    size = 32
    projector = FanBeam(img_size=size, n_angles=36, circle=False, backend=backend).to(device)
    centre = torch.zeros(1, 1, size, size, device=device)
    centre[..., size // 2, size // 2] = 1
    centre_sum = float(projector(centre).sum())
    for i, j in ((0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1), (1, 2)):
        x = torch.zeros(1, 1, size, size, device=device)
        x[..., i, j] = 1
        y = projector(x)
        assert float(y.sum()) > 0.5 * centre_sum, (i, j)
        back = projector.adjoint(torch.ones_like(y))
        torch.testing.assert_close(back[0, 0, i, j], y.sum(), rtol=1e-4, atol=1e-6)


def test_circle_still_masks_the_corners():
    size = 32
    projector = FanBeam(img_size=size, n_angles=36, circle=True)
    x = torch.zeros(1, 1, size, size)
    x[..., 0, 0] = 1
    assert float(projector(x).abs().sum()) == 0.0


def test_default_samples_follow_the_ray_radius():
    assert FanBeam(img_size=64, n_angles=4).n_samples == 64
    full = FanBeam(img_size=64, n_angles=4, circle=False)
    assert full.n_samples >= 64 * 2**0.5
    assert FanBeam(img_size=64, n_angles=4, circle=False, n_samples=50).n_samples == 50
