"""Tests for fan beam projector."""

import argparse
import sys
from pathlib import Path

import torch

from torchtomo import FanBeam, ParallelBeam, shepp_logan

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "ellipses"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))


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

    def test_subset_keeps_fan_geometry(self):
        from classical import subset_projectors

        parent = FanBeam(img_size=32, n_angles=12, n_samples=8)
        pairs = subset_projectors(parent, 3)
        assert len(pairs) == 3
        for indices, subset in pairs:
            assert subset.src_dist == parent.src_dist
            assert subset.det_dist == parent.det_dist
            assert subset.n_det == parent.n_det
            assert subset.n_samples == parent.n_samples
            torch.testing.assert_close(subset.angles, parent.angles[indices])

    def test_subset_parallel_does_not_grow_fan_kwargs(self):
        from classical import subset_projectors

        parent = ParallelBeam(img_size=32, n_angles=12)
        _, subset = subset_projectors(parent, 3)[0]
        assert not hasattr(subset, "src_dist")
        assert subset.n_det == parent.n_det

    def _assert_iradonmap_matches_fbp(self, projector):
        from models import IRadonMap

        model = IRadonMap(projector, width=4)
        y = torch.randn(2, 1, projector.n_angles, projector.n_det)
        with torch.no_grad():
            learned = model.sinusoidal_backprojection(y @ model.filtering.t())
            analytic = projector.fbp(y)
        torch.testing.assert_close(learned, analytic, rtol=1e-4, atol=1e-5)

    def test_iradonmap_untrained_matches_parallel_fbp(self):
        self._assert_iradonmap_matches_fbp(ParallelBeam(img_size=32, n_angles=16))

    def test_iradonmap_untrained_matches_fan_fbp(self):
        self._assert_iradonmap_matches_fbp(FanBeam(img_size=32, n_angles=16, n_samples=8))

    def test_compare_libraries_parallel_default_and_fan_span(self):
        benchmark = Path(__file__).resolve().parents[1] / "benchmark"
        if str(benchmark) not in sys.path:
            sys.path.insert(0, str(benchmark))
        from compare_libraries import angle_tensor, available_backends

        assert available_backends()[0].__name__ == "TorchtomoBackend"
        assert available_backends("fan")[0].__name__ == "TorchtomoFanBackend"
        parallel = angle_tensor(10, torch.device("cpu"), "parallel")
        fan = angle_tensor(10, torch.device("cpu"), "fan")
        torch.testing.assert_close(parallel[-1], torch.tensor(9 * torch.pi / 10))
        torch.testing.assert_close(fan[-1], torch.tensor(9 * 2 * torch.pi / 10))

    def test_train_build_projector_default_stays_parallel(self):
        from train import build_projector

        parallel_args = argparse.Namespace(
            geometry="parallel",
            projector="torchtomo",
            image_size=32,
            angles=16,
            src_dist=None,
            det_dist=None,
            n_det=None,
        )
        assert isinstance(build_projector(parallel_args), ParallelBeam)
        fan_args = argparse.Namespace(**{**parallel_args.__dict__, "geometry": "fan"})
        fan = build_projector(fan_args)
        assert isinstance(fan, FanBeam)
        assert fan.src_dist == 64
        assert fan.n_det == 48
