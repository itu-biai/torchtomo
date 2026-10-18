"""Independent transpose, physical integration, and differentiation checks for 3D rays."""

import pytest
import torch

from torchtomo import ConeBeam


def projector(chunk=5):
    sources = torch.tensor([[-7.1, 1.2, 0.4], [0.6, -8.3, 1.1]], dtype=torch.float64)
    detector = torch.tensor(
        [[[[7.3, -1.4, -0.2], [7.1, 0.8, 1.7]]], [[[0.3, 8.1, -0.7], [1.6, 8.2, 0.9]]]],
        dtype=torch.float64,
    )
    return ConeBeam((3, 4, 5), sources, detector, (1.3, 1.1, 0.9), (0.1, -0.2, 0.3), n_samples=37, ray_chunk=chunk)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_transpose_and_input_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(42)
    operator = projector().to(device)
    basis = torch.eye(60, dtype=torch.float64, device=device).reshape(60, 1, 3, 4, 5)
    matrix = operator(basis).flatten(1).T
    volume = torch.randn(2, 1, 3, 4, 5, dtype=torch.float64, device=device, requires_grad=True)
    projections = torch.randn(2, 1, 2, 1, 2, dtype=torch.float64, device=device, requires_grad=True)
    actual = operator.adjoint(projections)
    torch.testing.assert_close(actual, (projections.flatten(1) @ matrix).reshape_as(volume), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close((operator(volume) * projections).sum(), (volume * actual).sum())
    torch.testing.assert_close(torch.autograd.grad(operator(volume), volume, projections)[0], actual)
    torch.testing.assert_close(torch.autograd.grad(actual, projections, volume)[0], operator(volume))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_source_and_detector_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(12)
    operator = projector().to(device)
    source = operator.source_positions.clone().requires_grad_()
    detector = operator.detector_positions.clone().requires_grad_()
    volume = torch.randn(1, 1, 3, 4, 5, dtype=torch.float64, device=device)
    projections = torch.randn(1, 1, 2, 1, 2, dtype=torch.float64, device=device)

    def forward(sources, detectors):
        return ConeBeam((3, 4, 5), sources, detectors, (1.3, 1.1, 0.9), (0.1, -0.2, 0.3), n_samples=37)(volume)

    def adjoint(sources, detectors):
        return ConeBeam((3, 4, 5), sources, detectors, (1.3, 1.1, 0.9), (0.1, -0.2, 0.3), n_samples=37).adjoint(
            projections
        )

    assert torch.autograd.gradcheck(forward, (source, detector), atol=1e-5)
    assert torch.autograd.gradcheck(adjoint, (source, detector), atol=1e-5)


def test_axis_units_center_and_segment_clipping():
    source = torch.tensor([[-10.0, 2.0, 3.0]], dtype=torch.float64)
    detector = torch.tensor([[[[10.0, 2.0, 3.0], [-1.0, 2.0, 3.0], [-10.0, 2.0, 3.0]]]], dtype=torch.float64)
    operator = ConeBeam((3, 3, 5), source, detector, (2.0, 3.0, 1.0), (0.0, 2.0, 3.0), n_samples=1000)
    volume = torch.ones(1, 1, 3, 3, 5, dtype=torch.float64)
    expected = torch.tensor([[[[[4.75, 1.375, 0.0]]]]], dtype=torch.float64)
    torch.testing.assert_close(operator(volume), expected, atol=2e-5, rtol=0)


def test_missed_volume_and_chunk_invariance():
    torch.manual_seed(8)
    volume = torch.randn(2, 1, 3, 4, 5, dtype=torch.float64)
    torch.testing.assert_close(projector(chunk=1)(volume), projector(chunk=10)(volume))
    source = torch.tensor([[-10.0, 30.0, 0.0]], dtype=torch.float64)
    detector = torch.tensor([[[[10.0, 30.0, 0.0]]]], dtype=torch.float64)
    missed = ConeBeam((3, 4, 5), source, detector)
    assert torch.count_nonzero(missed(volume)) == 0
    assert torch.count_nonzero(missed.adjoint(torch.ones(1, 1, 1, 1, 1, dtype=torch.float64))) == 0


def test_input_validation_and_state_dict():
    operator = projector()
    with pytest.raises(ValueError, match="expected"):
        operator(torch.ones(1, 1, 3, 4, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="dtype"):
        operator(torch.ones(1, 1, 3, 4, 5))
    restored = projector()
    restored.load_state_dict(operator.state_dict())
    with torch.inference_mode():
        data = torch.ones(1, 1, 2, 1, 2, dtype=torch.float64)
        torch.testing.assert_close(restored.backward(data), operator.adjoint(data))
