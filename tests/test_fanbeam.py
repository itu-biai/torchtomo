"""Tests for fan beam projector."""

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
        assert projector.det_width == 200.0

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
