"""Tests for parallel beam projector."""

from torchtomo import ParallelBeam, shepp_logan


class TestParallelBeam:
    def test_forward_shape(self, parallel_projector, phantom_small):
        sinogram = parallel_projector.forward(phantom_small)
        B, C, n_angles, n_det = sinogram.shape
        assert B == 1
        assert C == 1
        assert n_angles == parallel_projector.n_angles
        assert n_det == parallel_projector.n_det

    def test_backward_shape(self, parallel_projector, phantom_small):
        sinogram = parallel_projector.forward(phantom_small)
        recon = parallel_projector.backward(sinogram)
        assert recon.shape == phantom_small.shape

    def test_fbp_shape(self, parallel_projector, phantom_small):
        sinogram = parallel_projector.forward(phantom_small)
        recon = parallel_projector.fbp(sinogram)
        assert recon.shape == phantom_small.shape

    def test_fbp_quality(self, img_size):
        projector = ParallelBeam(img_size=256, n_angles=180, n_det=256)
        phantom = shepp_logan(256)
        sinogram = projector.forward(phantom)
        recon = projector.fbp(sinogram)
        mse = ((recon - phantom) ** 2).mean().item()
        assert mse < 0.01

    def test_differentiable(self, parallel_projector, phantom_small):
        phantom = phantom_small.clone().requires_grad_(True)
        sinogram = parallel_projector.forward(phantom)
        loss = sinogram.sum()
        loss.backward()
        assert phantom.grad is not None
        assert phantom.grad.shape == phantom.shape

    def test_repr(self, parallel_projector):
        repr_str = repr(parallel_projector)
        assert "ParallelBeam" in repr_str
        assert "img_size" in repr_str
