"""Discrete transpose and training-gradient checks against explicit matrices."""

import pytest
import torch

from torchtomo import FanBeam, ParallelBeam


def _projector(geometry, size=5, circle=True):
    if geometry == "parallel":
        return ParallelBeam(img_size=size, n_angles=7, angle_range=(0.13, 2.71), circle=circle)
    return FanBeam(
        img_size=size,
        n_angles=7,
        n_det=size + 2,
        src_dist=2 * size,
        det_dist=3 * size,
        n_samples=13,
        angle_range=(0.13, 5.71),
        circle=circle,
    )


def _matrix(projector):
    """Materialize A from basis images, without using autograd or backward()."""
    size = projector.img_size
    basis = torch.eye(size**2, dtype=projector.angles.dtype, device=projector.angles.device)
    with torch.no_grad():
        return projector(basis.reshape(-1, 1, size, size)).flatten(1).T


@pytest.mark.parametrize("geometry", ["parallel", "fan"])
@pytest.mark.parametrize("circle", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_backward_matches_matrix_transpose(geometry, circle, dtype):
    torch.manual_seed(4)
    projector = _projector(geometry, circle=circle).to(dtype=dtype)
    matrix = _matrix(projector)
    y = torch.randn(3, 1, projector.n_angles, projector.n_det, dtype=dtype)
    # Exercise non-contiguous inputs and more than one image per batch.
    y = y.transpose(-1, -2).contiguous().transpose(-1, -2)
    expected = (y.flatten(1) @ matrix).reshape(3, 1, projector.img_size, projector.img_size)
    actual = projector.backward(y)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(projector.adjoint(y), actual)


@pytest.mark.parametrize("geometry", ["parallel", "fan"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inner_product_identity(geometry, dtype):
    torch.manual_seed(12)
    projector = _projector(geometry, size=16).to(dtype=dtype)
    x = torch.randn(8, 1, 16, 16, dtype=dtype)
    y = torch.randn(8, 1, projector.n_angles, projector.n_det, dtype=dtype)
    ax, aty = projector(x), projector.backward(y)
    lhs = (ax * y).flatten(1).sum(1)
    rhs = (x * aty).flatten(1).sum(1)
    # A norm-based denominator remains well-conditioned for nearly zero dots.
    scale = ax.flatten(1).norm(dim=1) * y.flatten(1).norm(dim=1)
    scale += x.flatten(1).norm(dim=1) * aty.flatten(1).norm(dim=1)
    assert torch.all((lhs - rhs).abs() <= 10 * torch.finfo(dtype).eps * scale)


@pytest.mark.parametrize("geometry", ["parallel", "fan"])
def test_both_operator_gradients(geometry):
    torch.manual_seed(5)
    projector = _projector(geometry).double()
    x = torch.randn(2, 1, 5, 5, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 1, projector.n_angles, projector.n_det, dtype=torch.float64, requires_grad=True)
    forward_vjp = torch.autograd.grad(projector(x), x, y)[0]
    backward_vjp = torch.autograd.grad(projector.backward(y), y, x)[0]
    torch.testing.assert_close(forward_vjp, projector.backward(y))
    torch.testing.assert_close(backward_vjp, projector(x))


@pytest.mark.parametrize("geometry", ["parallel", "fan"])
def test_backward_gradcheck(geometry):
    torch.manual_seed(6)
    projector = _projector(geometry, size=3, circle=False).double()
    y = torch.randn(1, 1, projector.n_angles, projector.n_det, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(projector.backward, (y,))
    assert torch.autograd.gradgradcheck(projector.backward, (y,))


@pytest.mark.parametrize("geometry", ["parallel", "fan"])
def test_unrolled_training_matches_explicit_matrix(geometry):
    """Check upstream parameter/data gradients through repeated A and A^T calls."""
    torch.manual_seed(7)
    projector = _projector(geometry).double()
    matrix = _matrix(projector)
    x = torch.randn(2, 1, 5, 5, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 1, projector.n_angles, projector.n_det, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[0.2, 0.1], [0.3, 0.15], [0.25, 0.2]], dtype=torch.float64, requires_grad=True)

    def unroll(forward, backward):
        primal, dual = x, torch.zeros_like(y)
        for dual_step, primal_step in weights:
            dual = torch.tanh(dual + dual_step * (forward(primal) - y))
            primal = torch.tanh(primal - primal_step * backward(dual))
        return primal

    actual = unroll(projector.forward, projector.backward)
    expected = unroll(
        lambda image: (image.flatten(1) @ matrix.T).reshape_as(y),
        lambda sino: (sino.flatten(1) @ matrix).reshape_as(x),
    )
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().sum(), (x, y, weights))
    expected_grads = torch.autograd.grad(expected.square().sum(), (x, y, weights))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        assert actual_grad.norm() > 0
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("geometry", ["parallel", "fan"])
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_backward_without_grad_tracking(geometry, context):
    projector = _projector(geometry).double()
    y = torch.ones(2, 1, projector.n_angles, projector.n_det, dtype=torch.float64)
    expected = projector.backward(y)
    with context():
        actual = projector.backward(y.clone())
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("geometry", ["parallel", "fan"])
def test_backward_with_angle_chunks(geometry, monkeypatch):
    torch.manual_seed(8)
    projector = _projector(geometry).double()
    y = torch.randn(2, 1, projector.n_angles, projector.n_det, dtype=torch.float64, requires_grad=True)
    expected = projector.backward(y)
    expected_grad = torch.autograd.grad(expected.square().sum(), y)[0]
    monkeypatch.setattr(projector, "_angle_chunk_size", lambda *args: 2)
    actual = projector.backward(y)
    actual_grad = torch.autograd.grad(actual.square().sum(), y)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("geometry", ["parallel", "fan"])
def test_fbp_uses_analytical_backprojection(geometry, monkeypatch):
    projector = _projector(geometry)
    y = torch.ones(1, 1, projector.n_angles, projector.n_det, requires_grad=True)

    def unexpected_adjoint(*args):
        pytest.fail("FBP must keep its analytical backprojection and normalization")

    monkeypatch.setattr(projector, "adjoint", unexpected_adjoint)
    # Both the preserved analytical operation and FBP must remain differentiable.
    for operation in (projector.backproject, projector.fbp):
        grad = torch.autograd.grad(operation(y).square().sum(), y)[0]
        assert torch.isfinite(grad).all()
        assert grad.norm() > 0


@pytest.mark.parametrize("geometry", ["parallel", "fan"])
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_adjoint_and_training(geometry, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS unavailable")
    torch.manual_seed(9)
    projector = _projector(geometry).to(device)
    matrix = _matrix(projector)
    y = torch.randn(2, 1, projector.n_angles, projector.n_det, device=device, requires_grad=True)
    expected = (y.flatten(1) @ matrix).reshape(2, 1, 5, 5)
    actual = projector.backward(y)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual_grad = torch.autograd.grad(actual.square().sum(), y)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), y)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)

    # Exercise training through both operators in a composed graph on device.
    x = torch.randn(2, 1, 5, 5, device=device, requires_grad=True)
    actual = projector.backward(projector(x) + y)
    expected = ((x.flatten(1) @ matrix.T + y.flatten(1)) @ matrix).reshape_as(x)
    actual_grads = torch.autograd.grad(actual.square().mean(), (x, y))
    expected_grads = torch.autograd.grad(expected.square().mean(), (x, y))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)
