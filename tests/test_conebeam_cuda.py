"""ConeBeam and BreastTomosynthesis on backend='cuda', against the PyTorch path and against themselves."""

import math

import pytest
import torch

from torchtomo import BreastTomosynthesis, ConeBeam
from torchtomo._nvrtc import runtime_available


def _require_kernels():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not runtime_available():
        pytest.skip("NVRTC or the CUDA driver unavailable")


def _tilted_circle(views=5, rows=6, columns=7):
    """A circular scan with a tilted, off-centre panel, around an off-centre anisotropic volume."""
    generator = torch.Generator().manual_seed(0)
    angles = torch.rand(views, generator=generator) * 2 * math.pi
    sources = torch.stack([30 * angles.cos(), 30 * angles.sin(), torch.randn(views, generator=generator)], -1)
    row = (torch.arange(rows) - (rows - 1) / 2) * 1.7
    column = (torch.arange(columns) - (columns - 1) / 2) * 1.9
    centres = -20 * torch.stack([angles.cos(), angles.sin(), torch.zeros(views)], -1)
    along = torch.stack([-angles.sin(), angles.cos(), torch.zeros(views)], -1)
    tilt = torch.tensor([0.1, 0.2, 1.0])
    detectors = centres[:, None, None] + row[:, None, None] * tilt + column[:, None] * along[:, None, None]
    return dict(
        volume_shape=(6, 7, 8),
        source_positions=sources,
        detector_positions=detectors,
        voxel_spacing=(1.3, 1.1, 0.9),
        volume_center=(0.4, -0.3, 0.2),
        n_samples=29,
    )


def _edge_rays():
    """Rays parallel to the axes, from a source inside the volume, to detectors inside and outside it, and one miss."""
    sources = torch.tensor([[0.3, 0.2, 0.1], [-20.0, 0.37, 0.21]])
    detectors = torch.tensor(
        [
            [
                [[1.1, -0.4, 0.6], [12.0, 3.0, -2.0], [0.3, 9.0, 0.1]],
                [[0.3, 0.2, -9.0], [0.31, 0.2, 0.1], [5.0, 5.0, 5.0]],
            ],
            [
                [[20.0, 0.37, 0.21], [20.0, 0.87, 0.21], [20.0, 30.0, 0.21]],
                [[2.0, 0.37, 0.21], [20.0, -1.0, 1.0], [9.0, 9.0, -9.0]],
            ],
        ]
    )
    return dict(
        volume_shape=(4, 5, 6),
        source_positions=sources,
        detector_positions=detectors,
        voxel_spacing=(1.0, 0.8, 1.2),
        volume_center=(0.1, -0.2, 0.3),
        n_samples=17,
    )


GEOMETRIES = [pytest.param(_tilted_circle, id="tilted-circle"), pytest.param(_edge_rays, id="edge-rays")]


def _pair(make, **overrides):
    arguments = {**make(), **overrides}
    arguments["source_positions"] = arguments["source_positions"].cuda()
    arguments["detector_positions"] = arguments["detector_positions"].cuda()
    return ConeBeam(**arguments), ConeBeam(**arguments, backend="cuda")


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5 * expected.abs().max().item())


@pytest.mark.parametrize("make", GEOMETRIES)
def test_matches_pytorch_path(make):
    _require_kernels()
    eager, fast = _pair(make)
    torch.manual_seed(1)
    for batch in (1, 2, 3, 5, 9):
        volume = torch.randn(batch, 1, *fast.volume_shape, device="cuda")
        projections = torch.randn(batch, 1, *fast.projection_shape, device="cuda")
        with torch.no_grad():
            _close(fast(volume), eager(volume))
            _close(fast.adjoint(projections), eager.adjoint(projections))
            torch.testing.assert_close(fast.backward(projections), fast.adjoint(projections))


@pytest.mark.parametrize("make", GEOMETRIES)
def test_adjoint_is_the_exact_transpose(make):
    """Every entry of A^T equals the matching entry of A; a missed or extra sample would show."""
    _require_kernels()
    _, fast = _pair(make)
    voxels, rays = math.prod(fast.volume_shape), math.prod(fast.projection_shape)
    with torch.no_grad():
        forward = fast(torch.eye(voxels, device="cuda").view(voxels, 1, *fast.volume_shape)).reshape(voxels, rays)
        basis = torch.eye(rays, device="cuda").view(rays, 1, *fast.projection_shape)
        adjoint = torch.cat([fast.adjoint(chunk) for chunk in basis.split(64)]).reshape(rays, voxels)
    assert forward.abs().max() > 0
    torch.testing.assert_close(adjoint, forward.t(), rtol=1e-5, atol=1e-6 * forward.abs().max().item())


def test_dot_product_at_96():
    _require_kernels()
    size, views = 96, 60
    angles = torch.arange(views) * (2 * math.pi / views)
    sources = torch.stack([4 * size * angles.cos(), 4 * size * angles.sin(), torch.zeros(views)], -1)
    row = column = (torch.arange(144) - 71.5) * 2.0
    centres = -4 * size * torch.stack([angles.cos(), angles.sin(), torch.zeros(views)], -1)
    along = torch.stack([-angles.sin(), angles.cos(), torch.zeros(views)], -1)
    axis = torch.tensor([0.0, 0.0, 1.0])
    detectors = centres[:, None, None] + row[:, None, None] * axis + column[:, None] * along[:, None, None]
    projector = ConeBeam((size,) * 3, sources.cuda(), detectors.cuda(), backend="cuda")
    torch.manual_seed(2)
    volume = torch.rand(5, 1, size, size, size, device="cuda")
    projections = torch.rand(5, 1, *projector.projection_shape, device="cuda")
    with torch.no_grad():
        lhs = (projector(volume).double() * projections.double()).sum()
        rhs = (volume.double() * projector.adjoint(projections).double()).sum()
    assert abs(lhs - rhs) / abs(lhs) < 1e-6


def test_gradients_are_the_matched_operators():
    _require_kernels()
    _, fast = _pair(_tilted_circle)
    torch.manual_seed(3)
    volume = torch.randn(3, 1, *fast.volume_shape, device="cuda", requires_grad=True)
    projections = torch.randn(3, 1, *fast.projection_shape, device="cuda", requires_grad=True)
    grad_projections = torch.randn_like(projections)
    grad_volume = torch.randn_like(volume)

    (gradient,) = torch.autograd.grad(fast(volume), volume, grad_projections)
    torch.testing.assert_close(gradient, fast.adjoint(grad_projections))
    (gradient,) = torch.autograd.grad(fast.adjoint(projections), projections, grad_volume)
    torch.testing.assert_close(gradient, fast(grad_volume))

    (gradient,) = torch.autograd.grad(fast(volume), volume, projections, create_graph=True)
    (second,) = torch.autograd.grad(gradient, projections, grad_volume)
    torch.testing.assert_close(second, fast(grad_volume))


@pytest.mark.parametrize("make", GEOMETRIES)
@pytest.mark.parametrize("operation", ["forward", "adjoint"])
def test_geometry_gradients_match_float64_autograd(make, operation):
    _require_kernels()
    arguments = make()
    torch.manual_seed(4)
    volume = torch.randn(3, 1, *arguments["volume_shape"], device="cuda")
    projections = torch.randn(3, 1, *arguments["detector_positions"].shape[:3], device="cuda")
    data, upstream = (volume, projections) if operation == "forward" else (projections, volume)

    def gradients(dtype, backend):
        sources = arguments["source_positions"].to("cuda", dtype).requires_grad_()
        detectors = arguments["detector_positions"].to("cuda", dtype).requires_grad_()
        settings = {key: value for key, value in arguments.items() if not key.endswith("positions")}
        projector = ConeBeam(source_positions=sources, detector_positions=detectors, backend=backend, **settings)
        apply = projector if operation == "forward" else projector.adjoint
        return torch.autograd.grad(apply(data.to(dtype)), (sources, detectors), upstream.to(dtype))

    for actual, expected in zip(gradients(torch.float32, "cuda"), gradients(torch.float64, "torch")):
        torch.testing.assert_close(actual.double(), expected, rtol=1e-4, atol=1e-5 * expected.abs().max().item())


def test_a_second_derivative_through_the_geometry_raises():
    _require_kernels()
    arguments = _tilted_circle()
    sources = arguments.pop("source_positions").cuda().requires_grad_()
    detectors = arguments.pop("detector_positions").cuda()
    projector = ConeBeam(source_positions=sources, detector_positions=detectors, backend="cuda", **arguments)
    volume = torch.randn(1, 1, *projector.volume_shape, device="cuda")
    (gradient,) = torch.autograd.grad(projector(volume).square().sum(), sources, create_graph=True)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), sources)


def test_other_tensors_fall_back_to_the_pytorch_path():
    arguments = _tilted_circle()
    reference = ConeBeam(**arguments)
    volume = torch.randn(2, 1, *reference.volume_shape)
    for backend in ("cuda", "auto"):
        projector = ConeBeam(**arguments, backend=backend)
        torch.testing.assert_close(projector(volume), reference(volume))
    assert ConeBeam(**arguments, backend="auto").backend in ("cuda", "torch")
    with pytest.raises(ValueError, match="backend"):
        ConeBeam(**arguments, backend="triton")
    if torch.cuda.is_available():
        double = {**arguments, "source_positions": arguments["source_positions"].cuda().double()}
        double["detector_positions"] = arguments["detector_positions"].cuda().double()
        eager, fast = ConeBeam(**double), ConeBeam(**double, backend="cuda")
        volume = volume.cuda().double()
        torch.testing.assert_close(fast(volume), eager(volume))


def _tomosynthesis(**kwargs):
    return BreastTomosynthesis(
        (8, 12, 16),
        angles=torch.linspace(-0.44, 0.44, 7),
        detector_shape=(20, 14),
        pixel_size=1.0,
        source_to_detector=40.0,
        source_to_axis=36.0,
        source_y=-6.0,
        voxel_spacing=(1.5, 1.0, 1.0),
        volume_center=(0.5, 0.0, -1.0),
        n_samples=20,
        device="cuda",
        **kwargs,
    )


def test_tomosynthesis_on_the_kernels():
    """A trainable arc gets the same gradient on the kernels as on the PyTorch path."""
    _require_kernels()
    torch.manual_seed(5)
    volume = torch.rand(2, 1, 8, 12, 16, device="cuda")
    measured = torch.rand(2, 1, 7, 20, 14, device="cuda")
    results = {}
    for backend in ("torch", "cuda"):
        operator = _tomosynthesis(learnable_geometry=True, backend=backend)
        loss = (operator(volume) - measured).square().sum() + (operator.adjoint(measured) * volume).sum()
        loss.backward()
        results[backend] = (loss.detach(), operator.angles.grad, operator.distances.grad)
    for actual, expected in zip(results["cuda"], results["torch"]):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4 * expected.abs().max().item())
