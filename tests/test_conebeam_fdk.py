"""ConeBeam.fdk and BreastTomosynthesis.fdk on both backends."""

import math

import pytest
import torch

from torchtomo import BreastTomosynthesis, ConeBeam
from torchtomo._fdk import fdk_geometry
from torchtomo._nvrtc import runtime_available


def _require_kernels():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not runtime_available():
        pytest.skip("NVRTC or the CUDA driver unavailable")


def _circle(size=24, views=72, device="cpu", dtype=torch.float32):
    """A full circular orbit, flat panel at magnification 2, rows along z."""
    angles = torch.arange(views, dtype=torch.float64) * (2 * math.pi / views)
    cosine, sine, zero = angles.cos(), angles.sin(), torch.zeros(views, dtype=torch.float64)
    sources = torch.stack([4 * size * cosine, 4 * size * sine, zero], dim=-1)
    centres = torch.stack([-4 * size * cosine, -4 * size * sine, zero], dim=-1)
    along = torch.stack([-sine, cosine, zero], dim=-1)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    pixels = (torch.arange(size * 3 // 2, dtype=torch.float64) - (size * 3 // 2 - 1) / 2) * 2
    detectors = centres[:, None, None] + pixels[:, None, None] * up + pixels[:, None] * along[:, None, None]
    options = {"device": device, "dtype": dtype}
    return dict(
        volume_shape=(size, size, size),
        source_positions=sources.to(**options),
        detector_positions=detectors.to(**options),
    )


def _ball(size, device="cpu", dtype=torch.float32):
    axis = (torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2) / (size / 2)
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    return ((x.square() + y.square() + z.square()) < 0.6**2).to(dtype)[None, None]


def _tomosynthesis(**kwargs):
    return BreastTomosynthesis(
        (8, 12, 16),
        angles=torch.linspace(-0.44, 0.44, 9),
        detector_shape=(20, 14),
        pixel_size=1.0,
        source_to_detector=40.0,
        source_to_axis=36.0,
        source_y=-6.0,
        voxel_spacing=(1.5, 1.0, 1.0),
        volume_center=(0.5, 0.0, -1.0),
        n_samples=20,
        **kwargs,
    )


def test_fdk_recovers_a_ball_at_its_value():
    """The scale is absolute: a unit ball comes back at 1 inside, 0 outside."""
    size = 24
    projector = ConeBeam(**_circle(size))
    ball = _ball(size)
    with torch.no_grad():
        volume = projector.fdk(projector(ball))
    axis = (torch.arange(size) - (size - 1) / 2) / (size / 2)
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    radius = (x.square() + y.square() + z.square()).sqrt()[None, None]
    assert abs(float(volume[radius < 0.4].mean()) - 1) < 0.02
    assert abs(float(volume[(radius > 0.8) & (radius < 0.95)].mean())) < 0.02


def test_filter_runs_along_source_motion():
    circle = _circle()
    geometry = fdk_geometry(
        circle["source_positions"], circle["detector_positions"], (24, 24, 24), torch.ones(3), torch.zeros(3)
    )
    assert geometry.axis == -1
    tomosynthesis = _tomosynthesis()
    operator = tomosynthesis.operator
    geometry = operator._fdk_geometry(operator.source_positions, operator.detector_positions)
    assert geometry.axis == -2


def test_fdk_rejects_what_it_cannot_reconstruct():
    circle = _circle()
    curved = circle["detector_positions"].clone()
    curved[..., 0] += 0.05 * torch.linspace(-1, 1, curved.shape[2]).square()
    projections = torch.zeros(1, 1, *curved.shape[:3])
    with pytest.raises(ValueError, match="flat"):
        ConeBeam(circle["volume_shape"], circle["source_positions"], curved).fdk(projections)
    single = ConeBeam(circle["volume_shape"], circle["source_positions"][:1], circle["detector_positions"][:1])
    with pytest.raises(ValueError, match="two views"):
        single.fdk(projections[:, :, :1])


def test_moving_the_geometry_in_place_refreshes_the_weights():
    circle = _circle(size=12, views=24)
    projector = ConeBeam(**circle)
    projections = torch.rand(1, 1, *projector.projection_shape)
    before = projector.fdk(projections)
    with torch.no_grad():
        projector.source_positions.mul_(1.5)
    after = projector.fdk(projections)
    fresh = ConeBeam(circle["volume_shape"], projector.source_positions.clone(), circle["detector_positions"])
    assert not torch.allclose(before, after)
    torch.testing.assert_close(after, fresh.fdk(projections))


def test_geometry_gradients_on_the_pytorch_path():
    circle = _circle(size=6, views=8, dtype=torch.float64)
    # Off the symmetric circle, so no voxel sits on a pixel boundary, a kink of bilinear interpolation.
    jitter = torch.randn(8, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    sources = (circle.pop("source_positions") + 0.3 * jitter).requires_grad_()
    detectors = circle.pop("detector_positions")
    projections = torch.rand(1, 1, *detectors.shape[:3], dtype=torch.float64)

    def reconstruct(sources):
        return ConeBeam(source_positions=sources, detector_positions=detectors, **circle).fdk(projections)

    assert torch.autograd.gradcheck(reconstruct, (sources,), eps=1e-6, atol=1e-6, rtol=1e-4)


def _cuda_pair(arguments):
    arguments = {
        **arguments,
        "source_positions": arguments["source_positions"].cuda(),
        "detector_positions": arguments["detector_positions"].cuda(),
    }
    return ConeBeam(**arguments), ConeBeam(**arguments, backend="cuda")


@pytest.mark.parametrize("filter_name", ["ramp", "hann"])
def test_kernels_match_the_pytorch_path(filter_name):
    _require_kernels()
    eager, fast = _cuda_pair(_circle(size=20, views=30))
    torch.manual_seed(0)
    for batch in (1, 2, 3, 5):
        projections = torch.randn(batch, 1, *fast.projection_shape, device="cuda")
        with torch.no_grad():
            expected = eager.fdk(projections, filter_name)
            torch.testing.assert_close(fast.fdk(projections, filter_name), expected, rtol=1e-4, atol=1e-5)


def test_kernel_gradients_are_the_transpose():
    _require_kernels()
    eager, fast = _cuda_pair(_circle(size=12, views=16))
    torch.manual_seed(1)
    projections = torch.randn(2, 1, *fast.projection_shape, device="cuda", requires_grad=True)
    upstream = torch.randn(2, 1, *fast.volume_shape, device="cuda", requires_grad=True)
    (expected,) = torch.autograd.grad(eager.fdk(projections), projections, upstream)
    (actual,) = torch.autograd.grad(fast.fdk(projections), projections, upstream, create_graph=True)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5 * expected.abs().max().item())
    # The weights are diagonal and the filter symmetric, so differentiating the
    # gradient again is FDK itself.
    direction = torch.randn_like(actual)
    (second,) = torch.autograd.grad(actual, upstream, direction)
    with torch.no_grad():
        torch.testing.assert_close(second, fast.fdk(direction), rtol=1e-4, atol=1e-5)


def test_tomosynthesis_fdk_on_both_backends():
    _require_kernels()
    torch.manual_seed(2)
    projections = torch.rand(2, 1, 9, 20, 14, device="cuda")
    eager = _tomosynthesis(device="cuda")
    fast = _tomosynthesis(device="cuda", backend="cuda")
    with torch.no_grad():
        torch.testing.assert_close(fast.fdk(projections, "hann"), eager.fdk(projections, "hann"))
    gradients = {}
    for backend in ("torch", "cuda"):
        operator = _tomosynthesis(device="cuda", backend=backend, learnable_geometry=True)
        operator.fdk(projections).square().sum().backward()
        gradients[backend] = (operator.angles.grad, operator.distances.grad)
    for actual, expected in zip(gradients["cuda"], gradients["torch"]):
        torch.testing.assert_close(actual, expected)
