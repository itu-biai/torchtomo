"""backend='cuda': runtime-compiled kernels against the PyTorch path and against themselves."""

import pytest
import torch

from torchtomo import FanBeam, ParallelBeam
from torchtomo._nvrtc import KernelLibrary, KernelRuntimeError, runtime_available


def _require_kernels():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not runtime_available():
        pytest.skip("NVRTC or the CUDA driver unavailable")


def _pair(cls, **kwargs):
    return cls(**kwargs).cuda(), cls(**kwargs, backend="cuda").cuda()


GEOMETRIES = [
    (ParallelBeam, dict(img_size=32, n_angles=19)),
    (ParallelBeam, dict(img_size=33, n_angles=12, circle=False)),
    (ParallelBeam, dict(img_size=64, n_angles=45, angle_range=(0.3, 2.9))),
    (FanBeam, dict(img_size=32, n_angles=19)),
    (FanBeam, dict(img_size=33, n_angles=12, circle=False)),
    (FanBeam, dict(img_size=40, n_angles=30, n_det=50, n_samples=64, src_dist=60.0, det_dist=40.0)),
    (FanBeam, dict(img_size=48, n_angles=24, n_samples=1)),
    # A source a few pixels from the image: the adjoint's exact-division branch.
    (FanBeam, dict(img_size=24, n_angles=16, src_dist=20.0, det_dist=10.0)),
]


def _ids(case):
    cls, kwargs = case
    return cls.__name__ + "-" + "-".join(f"{k}={v}" for k, v in kwargs.items())


@pytest.mark.parametrize("case", GEOMETRIES, ids=[_ids(case) for case in GEOMETRIES])
def test_matches_pytorch_path(case):
    _require_kernels()
    cls, kwargs = case
    eager, fast = _pair(cls, **kwargs)
    torch.manual_seed(0)
    for batch in (1, 2, 3, 5, 9):
        x = torch.randn(batch, 1, eager.img_size, eager.img_size, device="cuda")
        y = torch.randn(batch, 1, eager.n_angles, eager.n_det, device="cuda")
        with torch.no_grad():
            ref_f, ref_a, ref_b = eager.forward(x), eager.adjoint(y), eager.backproject(y)
            f, a, b = fast.forward(x), fast.adjoint(y), fast.backproject(y)
        for got, ref in ((f, ref_f), (a, ref_a), (b, ref_b)):
            torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4 * ref.abs().max().item())
        torch.testing.assert_close(fast.backward(y), a)


@pytest.mark.parametrize("case", GEOMETRIES, ids=[_ids(case) for case in GEOMETRIES])
def test_adjoint_is_the_exact_transpose(case):
    """Every matrix entry of A^T equals the matching entry of A; a missed sample would show."""
    _require_kernels()
    cls, kwargs = case
    kwargs = dict(kwargs, img_size=min(kwargs["img_size"], 12), n_angles=min(kwargs["n_angles"], 7))
    if "n_det" in kwargs:
        kwargs["n_det"] = 15
    projector = cls(**kwargs, backend="cuda").cuda()
    size, n_rows = projector.img_size, projector.n_angles * projector.n_det
    images = torch.eye(size * size, device="cuda").view(-1, 1, size, size)
    sinograms = torch.eye(n_rows, device="cuda").view(-1, 1, projector.n_angles, projector.n_det)
    with torch.no_grad():
        forward = torch.cat([projector.forward(chunk) for chunk in images.split(64)]).reshape(size * size, n_rows)
        adjoint = torch.cat([projector.adjoint(chunk) for chunk in sinograms.split(64)]).reshape(n_rows, size * size)
    assert forward.abs().max() > 0
    torch.testing.assert_close(adjoint, forward.t(), rtol=1e-5, atol=1e-6 * forward.abs().max().item())


@pytest.mark.parametrize("cls", [ParallelBeam, FanBeam])
def test_dot_product_at_512(cls):
    _require_kernels()
    projector = cls(img_size=512, n_angles=90, backend="cuda").cuda()
    torch.manual_seed(1)
    x = torch.rand(5, 1, 512, 512, device="cuda")
    y = torch.rand(5, 1, 90, projector.n_det, device="cuda")
    with torch.no_grad():
        lhs = (projector.forward(x).double() * y.double()).sum()
        rhs = (x.double() * projector.adjoint(y).double()).sum()
    assert abs(lhs - rhs) / abs(lhs) < 1e-6


def test_fan_forward_is_as_accurate_as_the_pytorch_path():
    """The kernel's ray table is built in float64, so it sits at least as close to float64."""
    _require_kernels()
    kwargs = dict(img_size=256, n_angles=60)
    reference = FanBeam(**kwargs).cuda().double()
    eager, fast = _pair(FanBeam, **kwargs)
    torch.manual_seed(2)
    x = torch.rand(2, 1, 256, 256, device="cuda")
    with torch.no_grad():
        truth = reference.forward(x.double())
        eager_error = (eager.forward(x).double() - truth).abs().max()
        fast_error = (fast.forward(x).double() - truth).abs().max()
    assert fast_error <= eager_error * 1.01


@pytest.mark.parametrize("cls", [ParallelBeam, FanBeam])
def test_gradients_are_the_matched_operators(cls):
    _require_kernels()
    projector = cls(img_size=24, n_angles=10, backend="cuda").cuda()
    torch.manual_seed(3)
    x = torch.randn(3, 1, 24, 24, device="cuda", requires_grad=True)
    y = torch.randn(3, 1, 10, projector.n_det, device="cuda", requires_grad=True)
    g_sino = torch.randn(3, 1, 10, projector.n_det, device="cuda")
    g_image = torch.randn(3, 1, 24, 24, device="cuda")

    (grad_x,) = torch.autograd.grad(projector.forward(x), x, g_sino)
    torch.testing.assert_close(grad_x, projector.adjoint(g_sino))
    (grad_y,) = torch.autograd.grad(projector.adjoint(y), y, g_image)
    torch.testing.assert_close(grad_y, projector.forward(g_image))

    # Double backward: the VJP of the forward is itself differentiable.
    (grad_x,) = torch.autograd.grad(projector.forward(x), x, y, create_graph=True)
    (grad_y,) = torch.autograd.grad(grad_x, y, g_image)
    torch.testing.assert_close(grad_y, projector.forward(g_image))


@pytest.mark.parametrize("cls", [ParallelBeam, FanBeam])
def test_backproject_gradient_uses_the_pytorch_path(cls):
    _require_kernels()
    eager, fast = _pair(cls, img_size=16, n_angles=8)
    y = torch.randn(2, 1, 8, eager.n_det, device="cuda", requires_grad=True)
    g = torch.randn(2, 1, 16, 16, device="cuda")
    (fast_grad,) = torch.autograd.grad(fast.backproject(y), y, g)
    (eager_grad,) = torch.autograd.grad(eager.backproject(y), y, g)
    torch.testing.assert_close(fast_grad, eager_grad)


@pytest.mark.parametrize("cls", [ParallelBeam, FanBeam])
def test_unrolled_training_step(cls):
    """An LPD-style unroll: forward and adjoint inside the graph, a parameter learns."""
    _require_kernels()
    projector = cls(img_size=32, n_angles=12, backend="cuda").cuda()
    step = torch.nn.Parameter(torch.tensor(0.1, device="cuda"))
    x_true = torch.rand(2, 1, 32, 32, device="cuda")
    with torch.inference_mode():
        # Tables built under inference_mode must still be usable for training.
        y = projector.forward(x_true)
    y = y.clone()
    x = torch.zeros_like(x_true)
    for _ in range(3):
        x = x - step * projector.adjoint(projector.forward(x) - y)
    loss = (x - x_true).square().mean()
    loss.backward()
    assert step.grad is not None and torch.isfinite(step.grad) and step.grad != 0


def test_falls_back_on_cpu_and_float64():
    _require_kernels()
    for cls in (ParallelBeam, FanBeam):
        eager = cls(img_size=16, n_angles=6)
        fast = cls(img_size=16, n_angles=6, backend="cuda")
        x = torch.rand(2, 1, 16, 16)
        torch.testing.assert_close(fast.forward(x), eager.forward(x))
        fast64, eager64 = fast.cuda().double(), eager.cuda().double()
        x64 = x.cuda().double()
        torch.testing.assert_close(fast64.forward(x64), eager64.forward(x64))
        torch.testing.assert_close(fast64.adjoint(fast64.forward(x64)), eager64.adjoint(eager64.forward(x64)))


@pytest.mark.parametrize("backend", ["torch", "cuda", "auto"])
def test_fan_geometry_is_built_on_first_use(backend):
    """No backend pays for the PyTorch path's grids until something reads them."""
    fast = FanBeam(img_size=64, n_angles=30, backend=backend)
    assert fast._buffers["ray_grids"] is None and fast._buffers["backward_grids"] is None
    assert "ray_grids" not in fast.state_dict()
    # Reading one builds them, so code that inspects the geometry keeps working.
    assert fast.backward_weights.shape == (30, 64, 64)
    assert fast._buffers["ray_grids"] is not None


def test_backend_is_validated():
    with pytest.raises(ValueError):
        ParallelBeam(img_size=8, n_angles=4, backend="opencl")
    with pytest.raises(ValueError):
        FanBeam(img_size=8, n_angles=4, backend="triton")
    with pytest.raises(ValueError):
        ParallelBeam(img_size=8, n_angles=4, triton=True, backend="cuda")
    assert ParallelBeam(img_size=8, n_angles=4, triton=True).backend == "triton"


@pytest.mark.parametrize("cls", [ParallelBeam, FanBeam])
def test_auto_backend_follows_the_runtime(cls):
    projector = cls(img_size=8, n_angles=4, backend="auto")
    expected = "cuda" if runtime_available() else "torch"
    assert projector.backend == expected
    # It resolves to a real backend, so nothing downstream has to know about "auto".
    assert projector.backend in ("cuda", "torch")


@pytest.mark.parametrize("cls", [ParallelBeam, FanBeam])
def test_auto_backend_runs_the_same_geometry(cls):
    _require_kernels()
    auto = cls(img_size=32, n_angles=12, backend="auto").cuda()
    assert auto.backend == "cuda"
    x = torch.rand(1, 1, 32, 32, device="cuda")
    torch.testing.assert_close(auto.forward(x), cls(img_size=32, n_angles=12, backend="cuda").cuda().forward(x))


def test_compile_errors_carry_the_nvrtc_log():
    _require_kernels()
    library = KernelLibrary('extern "C" __global__ void broken() { undefined_symbol(); }', "broken.cu")
    with pytest.raises(KernelRuntimeError, match="undefined_symbol"):
        library.function("broken", torch.device("cuda"))


def test_compiled_kernels_are_cached_on_disk(tmp_path, monkeypatch):
    _require_kernels()
    monkeypatch.setenv("TORCHTOMO_KERNEL_CACHE", str(tmp_path))
    source = 'extern "C" __global__ void fill(float* out, float value) { out[threadIdx.x] = value; }'
    first = KernelLibrary(source, "fill.cu")
    out = torch.zeros(32, device="cuda")
    first.function("fill", out.device)((1, 1, 1), (32, 1, 1), [out, 2.5])
    assert first.last_from_cache is False
    assert len(list(tmp_path.iterdir())) == 1
    second = KernelLibrary(source, "fill.cu")
    second.function("fill", out.device)((1, 1, 1), (32, 1, 1), [out, 4.0])
    assert second.last_from_cache is True
    torch.cuda.synchronize()
    assert torch.all(out == 4.0)


@pytest.mark.parametrize("cls", [ParallelBeam, FanBeam])
def test_approximate_mode_is_close_to_the_exact_pair(cls):
    """Texture forward within 1e-3 of the exact one; pixel-driven adjoint within a few percent."""
    _require_kernels()
    from torchtomo import shepp_logan

    exact = cls(img_size=128, n_angles=60, backend="cuda").cuda()
    approx = cls(img_size=128, n_angles=60, backend="cuda", approximate=True).cuda()
    phantom = shepp_logan(128, device="cuda")
    for batch in (1, 2, 3, 5):
        x = phantom.repeat(batch, 1, 1, 1) * torch.linspace(0.5, 1.5, batch, device="cuda").view(-1, 1, 1, 1)
        with torch.no_grad():
            f_exact, f_approx = exact.forward(x), approx.forward(x)
            # A smooth sinogram, where interpolation models agree up to their kernels.
            a_exact, a_approx = exact.adjoint(f_exact), approx.adjoint(f_exact)
        assert ((f_approx - f_exact).norm() / f_exact.norm()) < 1e-3
        assert ((a_approx - a_exact).norm() / a_exact.norm()) < 0.03
        for k in range(batch):
            assert ((f_approx[k] - f_exact[k]).norm() / f_exact[k].norm()) < 1e-3


@pytest.mark.parametrize("cls", [ParallelBeam, FanBeam])
def test_approximate_mode_gradients(cls):
    _require_kernels()
    projector = cls(img_size=32, n_angles=12, backend="cuda", approximate=True).cuda()
    x = torch.rand(3, 1, 32, 32, device="cuda", requires_grad=True)
    g = torch.randn(3, 1, 12, projector.n_det, device="cuda")
    (grad,) = torch.autograd.grad(projector.forward(x), x, g)
    torch.testing.assert_close(grad, projector.adjoint(g))


def test_approximate_mode_is_validated():
    with pytest.raises(ValueError):
        ParallelBeam(img_size=8, n_angles=4, approximate=True)
    with pytest.raises(ValueError):
        FanBeam(img_size=8, n_angles=4, approximate=True)
    with pytest.raises(ValueError):
        FanBeam(img_size=8, n_angles=4, backend="cuda", approximate=True, n_samples=1)
    # "auto" may land on the PyTorch path, which ignores it rather than refusing it.
    assert ParallelBeam(img_size=8, n_angles=4, backend="auto", approximate=True).approximate


def _shift(*projectors):
    """The same per-view offsets on each projector: every view moved, by a different amount."""
    for projector in projectors:
        n = projector.n_angles
        shifts = dict(detector_shift=torch.linspace(-2.5, 3.5, n))
        if "source_shift" in projector._POSE_COLUMNS:
            shifts["source_shift"] = torch.linspace(1.5, -1.0, n)
        projector.set_pose(**shifts)


@pytest.mark.parametrize("case", GEOMETRIES, ids=[_ids(case) for case in GEOMETRIES])
def test_shifted_pose_matches_pytorch_path(case):
    """The kernels read the shifts from their pose table, and stay on the kernels."""
    _require_kernels()
    cls, kwargs = case
    eager, fast = _pair(cls, **kwargs)
    _shift(eager, fast)
    torch.manual_seed(4)
    x = torch.randn(3, 1, eager.img_size, eager.img_size, device="cuda")
    y = torch.randn(3, 1, eager.n_angles, eager.n_det, device="cuda")
    assert fast._use_kernels(x)
    with torch.no_grad():
        pairs = (
            (fast.forward(x), eager.forward(x)),
            (fast.adjoint(y), eager.adjoint(y)),
            (fast.backproject(y), eager.backproject(y)),
        )
    for got, ref in pairs:
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4 * ref.abs().max().item())


@pytest.mark.parametrize(
    "case", GEOMETRIES[:2] + GEOMETRIES[3:5], ids=[_ids(c) for c in GEOMETRIES[:2] + GEOMETRIES[3:5]]
)
def test_shifted_adjoint_is_the_exact_transpose(case):
    _require_kernels()
    cls, kwargs = case
    kwargs = dict(kwargs, img_size=12, n_angles=7)
    projector = cls(**kwargs, backend="cuda").cuda()
    _shift(projector)
    size, n_rows = projector.img_size, projector.n_angles * projector.n_det
    images = torch.eye(size * size, device="cuda").view(-1, 1, size, size)
    sinograms = torch.eye(n_rows, device="cuda").view(-1, 1, projector.n_angles, projector.n_det)
    with torch.no_grad():
        forward = torch.cat([projector.forward(chunk) for chunk in images.split(64)]).reshape(size * size, n_rows)
        adjoint = torch.cat([projector.adjoint(chunk) for chunk in sinograms.split(64)]).reshape(n_rows, size * size)
    assert forward.abs().max() > 0
    torch.testing.assert_close(adjoint, forward.t(), rtol=1e-5, atol=1e-6 * forward.abs().max().item())


@pytest.mark.parametrize("cls", [ParallelBeam, FanBeam])
def test_shifted_approximate_mode_is_close_to_the_exact_pair(cls):
    _require_kernels()
    from torchtomo import shepp_logan

    exact = cls(img_size=128, n_angles=60, backend="cuda").cuda()
    approx = cls(img_size=128, n_angles=60, backend="cuda", approximate=True).cuda()
    _shift(exact, approx)
    x = shepp_logan(128, device="cuda")
    with torch.no_grad():
        f_exact, f_approx = exact.forward(x), approx.forward(x)
        a_exact, a_approx = exact.adjoint(f_exact), approx.adjoint(f_exact)
    assert ((f_approx - f_exact).norm() / f_exact.norm()) < 1e-3
    assert ((a_approx - a_exact).norm() / a_exact.norm()) < 0.03


@pytest.mark.parametrize("cls", [ParallelBeam, FanBeam])
def test_geometry_gradient_leaves_the_kernels(cls):
    """A pose that wants a gradient runs on the PyTorch path, and gets the same answer."""
    _require_kernels()
    eager, fast = _pair(cls, img_size=24, n_angles=10)
    x = torch.rand(1, 1, 24, 24, device="cuda")
    grads = []
    for projector in (eager, fast):
        projector.pose.requires_grad_(True)
        assert not projector._use_kernels(x)
        (grad,) = torch.autograd.grad(projector.forward(x).square().sum(), projector.pose)
        grads.append(grad)
    torch.testing.assert_close(grads[1], grads[0])


def test_triton_leaves_a_shifted_pose():
    """Triton reads angles only, so a shifted pose leaves it."""
    projector = ParallelBeam(img_size=16, n_angles=4, backend="triton")
    projector.set_pose(detector_shift=1.0)
    assert not projector._use_triton(torch.zeros(1, 1, 16, 16))
