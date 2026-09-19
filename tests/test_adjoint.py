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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_direct_adjoint_matches_vjp(device):
    """Shipped adjoint/backward match the VJP fallback, including a non-contiguous sinogram."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from torchtomo.base import _vjp_adjoint

    torch.manual_seed(21)
    projector = ParallelBeam(img_size=16, n_angles=11, angle_range=(0.13, 2.71), circle=True).double().to(device)
    y = torch.randn(3, 1, projector.n_angles, projector.n_det, dtype=torch.float64, device=device)
    y = y.transpose(-1, -2).contiguous().transpose(-1, -2)
    expected = _vjp_adjoint(projector, y)
    actual = projector.adjoint(y)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(projector.backward(y), actual)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adjoint_calls_direct_kernel(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import torchtomo.parallel as parallel_mod

    calls = {"n": 0}
    original = parallel_mod.grid_sample_input_backward

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(parallel_mod, "grid_sample_input_backward", wrapped)
    projector = ParallelBeam(img_size=8, n_angles=6, circle=False).to(device)
    y = torch.randn(2, 1, projector.n_angles, projector.n_det, device=device)
    projector.adjoint(y)
    assert calls["n"] > 0


def test_vjp_fallback_when_kernel_is_missing(monkeypatch):
    import torchtomo.base as base_mod
    from torchtomo.base import _vjp_adjoint

    monkeypatch.setattr(base_mod, "grid_sample_input_backward_supported", lambda device: False)
    torch.manual_seed(22)
    projector = ParallelBeam(img_size=8, n_angles=5, circle=False).double()
    y = torch.randn(2, 1, projector.n_angles, projector.n_det, dtype=torch.float64)
    torch.testing.assert_close(projector.adjoint(y), _vjp_adjoint(projector, y))


def test_fan_direct_adjoint_matches_vjp():
    from torchtomo.base import _vjp_adjoint

    torch.manual_seed(33)
    projector = _projector("fan", size=8).double()
    y = torch.randn(2, 1, projector.n_angles, projector.n_det, dtype=torch.float64)
    y = y.transpose(-1, -2).contiguous().transpose(-1, -2)
    expected = _vjp_adjoint(projector, y)
    actual = projector.adjoint(y)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(projector.backward(y), actual)
    x = torch.randn(2, 1, 8, 8, dtype=torch.float64)
    ax, aty = projector.forward(x), projector.adjoint(y)
    lhs = (ax * y).flatten(1).sum(1)
    rhs = (x * aty).flatten(1).sum(1)
    scale = ax.flatten(1).norm(dim=1) * y.flatten(1).norm(dim=1)
    scale = scale + x.flatten(1).norm(dim=1) * aty.flatten(1).norm(dim=1)
    assert torch.all((lhs - rhs).abs() <= 10 * torch.finfo(torch.float64).eps * scale)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fan_adjoint_calls_direct_kernel(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import torchtomo.fanbeam as fan_mod

    calls = {"n": 0}
    original = fan_mod.grid_sample_input_backward

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fan_mod, "grid_sample_input_backward", wrapped)
    projector = _projector("fan", size=5).to(device)
    y = torch.randn(1, 1, projector.n_angles, projector.n_det, device=device)
    projector.adjoint(y)
    assert calls["n"] > 0


def test_sparse_adjoint_matches_eager_and_inner_product():
    """Opt-in CSR adjoint matches the eager adjoint and the shipped inner-product identity."""
    from torchtomo.base import _vjp_adjoint

    torch.manual_seed(23)
    eager = ParallelBeam(img_size=16, n_angles=11, angle_range=(0.13, 2.71), circle=True).double()
    sparse = ParallelBeam(img_size=16, n_angles=11, angle_range=(0.13, 2.71), circle=True, sparse_adjoint=True).double()
    x = torch.randn(3, 1, 16, 16, dtype=torch.float64)
    y = torch.randn(3, 1, sparse.n_angles, sparse.n_det, dtype=torch.float64)
    y = y.transpose(-1, -2).contiguous().transpose(-1, -2)
    actual = sparse.adjoint(y)
    torch.testing.assert_close(actual, eager.adjoint(y))
    torch.testing.assert_close(actual, _vjp_adjoint(eager, y))
    torch.testing.assert_close(sparse.backward(y), actual)
    ax, aty = sparse.forward(x), sparse.adjoint(y)
    lhs = (ax * y).flatten(1).sum(1)
    rhs = (x * aty).flatten(1).sum(1)
    scale = ax.flatten(1).norm(dim=1) * y.flatten(1).norm(dim=1)
    scale = scale + x.flatten(1).norm(dim=1) * aty.flatten(1).norm(dim=1)
    assert torch.all((lhs - rhs).abs() <= 10 * torch.finfo(torch.float64).eps * scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_sparse_adjoint_matches_eager_cuda():
    torch.manual_seed(24)
    eager = ParallelBeam(img_size=8, n_angles=6, circle=False).to("cuda")
    sparse = ParallelBeam(img_size=8, n_angles=6, circle=False, sparse_adjoint=True).to("cuda")
    y = torch.randn(2, 1, 6, 8, device="cuda")
    torch.testing.assert_close(sparse.adjoint(y), eager.adjoint(y), rtol=2e-5, atol=2e-6)


def _require_triton_cuda():
    from torchtomo._triton_kernels import triton_kernels_available

    if not torch.cuda.is_available() or not triton_kernels_available(torch.device("cuda"), torch.float32):
        pytest.skip("Triton CUDA float32 unavailable")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_matches_eager_forward_adjoint_backproject():
    _require_triton_cuda()
    torch.manual_seed(26)
    for size, n_angles, circle in ((16, 7, True), (32, 45, False)):
        eager = ParallelBeam(img_size=size, n_angles=n_angles, circle=circle).cuda()
        fast = ParallelBeam(img_size=size, n_angles=n_angles, circle=circle, triton=True).cuda()
        x = torch.randn(2, 1, size, size, device="cuda")
        y = torch.randn(2, 1, n_angles, size, device="cuda")
        torch.testing.assert_close(fast.forward(x), eager.forward(x), rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(fast.adjoint(y), eager.adjoint(y), rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(fast.backward(y), fast.adjoint(y))
        torch.testing.assert_close(fast.backproject(y), eager.backproject(y), rtol=2e-4, atol=2e-5)
        ax, aty = fast.forward(x), fast.adjoint(y)
        lhs = (ax * y).flatten(1).sum(1)
        rhs = (x * aty).flatten(1).sum(1)
        scale = ax.flatten(1).norm(dim=1) * y.flatten(1).norm(dim=1)
        scale = scale + x.flatten(1).norm(dim=1) * aty.flatten(1).norm(dim=1)
        assert torch.all((lhs - rhs).abs() <= 5e-5 * scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_forward_vjp_matches_adjoint():
    """Shipped CUDA triton forward is differentiable; its VJP is the gather adjoint."""
    _require_triton_cuda()
    torch.manual_seed(28)
    projector = ParallelBeam(img_size=16, n_angles=11, circle=True, triton=True).cuda()
    x = torch.randn(2, 1, 16, 16, device="cuda", requires_grad=True)
    y = torch.randn(2, 1, 11, 16, device="cuda", requires_grad=True)
    sino = projector.forward(x)
    assert sino.grad_fn is not None
    assert "TritonProject" in type(sino.grad_fn).__name__
    from_forward = torch.autograd.grad(sino, x, y)[0]
    torch.testing.assert_close(from_forward, projector.adjoint(y), rtol=2e-4, atol=2e-5)
    aty = projector.backward(y)
    assert "TritonAdjoint" in type(aty.grad_fn).__name__
    from_backward = torch.autograd.grad(aty, y, x)[0]
    torch.testing.assert_close(from_backward, projector.forward(x.detach()), rtol=2e-4, atol=2e-5)
    x_loop = x.detach().requires_grad_(True)
    projector.adjoint(projector.forward(x_loop)).sum().backward()
    assert x_loop.grad is not None
    assert torch.isfinite(x_loop.grad).all()
    assert x_loop.grad.norm() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_adjoint_vjp_is_fused_forward():
    """y.requires_grad; VJP of backward is fused forward; A^T A x is a valid graph."""
    _require_triton_cuda()
    torch.manual_seed(32)
    projector = ParallelBeam(img_size=16, n_angles=11, circle=True, triton=True).cuda()
    x = torch.randn(2, 1, 16, 16, device="cuda")
    y = torch.randn(2, 1, 11, 16, device="cuda", requires_grad=True)
    from_adj = torch.autograd.grad(projector.backward(y), y, x)[0]
    torch.testing.assert_close(from_adj, projector.forward(x), rtol=2e-4, atol=2e-5)
    x_loop = x.detach().requires_grad_(True)
    projector.adjoint(projector.forward(x_loop)).sum().backward()
    assert x_loop.grad is not None and torch.isfinite(x_loop.grad).all() and x_loop.grad.norm() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_backproject_backward_succeeds():
    _require_triton_cuda()
    torch.manual_seed(29)
    projector = ParallelBeam(img_size=16, n_angles=11, circle=True, triton=True).cuda()
    y = torch.randn(2, 1, 11, 16, device="cuda", requires_grad=True)
    recon = projector.backproject(y)
    assert recon.grad_fn is not None
    grad = torch.autograd.grad(recon, y, torch.ones_like(recon))[0]
    assert torch.isfinite(grad).all()
    assert grad.norm() > 0
    eager = ParallelBeam(img_size=16, n_angles=11, circle=True).cuda()
    y_e = y.detach().requires_grad_(True)
    recon_e = eager.backproject(y_e)
    grad_e = torch.autograd.grad(recon_e, y_e, torch.ones_like(recon_e))[0]
    torch.testing.assert_close(grad, grad_e, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_double_falls_back_to_eager():
    _require_triton_cuda()
    torch.manual_seed(30)
    eager = ParallelBeam(img_size=8, n_angles=6, circle=False).double().cuda()
    fast = ParallelBeam(img_size=8, n_angles=6, circle=False, triton=True).double().cuda()
    x = torch.randn(2, 1, 8, 8, dtype=torch.float64, device="cuda", requires_grad=True)
    out = fast.forward(x)
    torch.testing.assert_close(out, eager.forward(x.detach()))
    grad = torch.autograd.grad(out.sum(), x)[0]
    assert torch.isfinite(grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triton_adjoint_gradcheck():
    _require_triton_cuda()
    torch.manual_seed(31)
    projector = ParallelBeam(img_size=3, n_angles=5, circle=False, triton=True).double().cuda()
    y = torch.randn(1, 1, projector.n_angles, projector.n_det, dtype=torch.float64, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(projector.backward, (y,))


def test_triton_flag_falls_back_on_cpu():
    torch.manual_seed(27)
    eager = ParallelBeam(img_size=8, n_angles=5, circle=False)
    flagged = ParallelBeam(img_size=8, n_angles=5, circle=False, triton=True)
    x = torch.randn(1, 1, 8, 8)
    y = torch.randn(1, 1, 5, 8)
    torch.testing.assert_close(flagged.forward(x), eager.forward(x))
    torch.testing.assert_close(flagged.adjoint(y), eager.adjoint(y))


def test_sparse_adjoint_gradcheck():
    torch.manual_seed(25)
    projector = ParallelBeam(img_size=3, n_angles=5, circle=False, sparse_adjoint=True).double()
    y = torch.randn(1, 1, projector.n_angles, projector.n_det, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(projector.backward, (y,))


def test_default_angles_exclude_the_far_endpoint_and_match_angle_step():
    projector = ParallelBeam(img_size=8, n_angles=4)
    expected = torch.arange(4, dtype=torch.float32) * (torch.pi / 4)
    torch.testing.assert_close(projector.angles, expected, atol=0, rtol=0)
    assert projector.angles[-1] < torch.pi
    spacing = projector.angles[1:] - projector.angles[:-1]
    torch.testing.assert_close(spacing, torch.full_like(spacing, projector.angle_step))


def test_explicit_angles_are_kept_and_cast_with_dtype():
    chosen = torch.tensor([0.1, 0.5, 1.2], dtype=torch.float64)
    projector = ParallelBeam(img_size=8, n_angles=3, angles=chosen)
    torch.testing.assert_close(projector.angles, chosen.float())
    projector64 = projector.double()
    torch.testing.assert_close(projector64.angles, chosen)
    fan = FanBeam(img_size=8, n_angles=3, n_det=10, n_samples=8, angles=chosen)
    torch.testing.assert_close(fan.angles, chosen.float())


def test_float64_coordinates_are_rebuilt_not_promoted():
    size, n_angles = 8, 7
    angle_range = (0.13, 2.71)
    projector = ParallelBeam(img_size=size, n_angles=n_angles, angle_range=angle_range)
    step = (angle_range[1] - angle_range[0]) / n_angles
    expected32 = torch.arange(n_angles, dtype=torch.float32) * step + angle_range[0]
    torch.testing.assert_close(projector.angles, expected32, atol=0, rtol=0)

    projector64 = projector.double()
    expected64 = torch.arange(n_angles, dtype=torch.float64) * step + angle_range[0]
    torch.testing.assert_close(projector64.angles, expected64, atol=0, rtol=0)
    coords = torch.linspace(-1, 1, size, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    torch.testing.assert_close(projector64.grid_x, grid_x, atol=0, rtol=0)
    torch.testing.assert_close(projector64.grid_y, grid_y, atol=0, rtol=0)


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
