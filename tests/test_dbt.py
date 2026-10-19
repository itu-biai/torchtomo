"""Arc geometry, stationary detector, and matched transpose for breast tomosynthesis."""

import copy
import math

import pytest
import torch

from torchtomo import BreastTomosynthesis
from torchtomo.dbt import arc_positions


def projector(**kwargs):
    angles = torch.linspace(math.radians(-25), math.radians(25), 5, dtype=torch.float64)
    defaults = dict(
        volume_shape=(4, 5, 6),
        angles=angles,
        detector_shape=(3, 4),
        pixel_size=0.5,
        source_to_detector=10.0,
        source_to_axis=8.0,
        voxel_spacing=(1.0, 0.7, 0.8),
        volume_center=(0.2, -0.3, 1.0),
        n_samples=11,
        ray_chunk=7,
        dtype=torch.float64,
    )
    defaults.update(kwargs)
    return BreastTomosynthesis(**defaults)


def test_stationary_detector_and_source_arc():
    operator = projector()
    detector = operator.detector_positions
    torch.testing.assert_close(detector[0], detector[-1])
    assert torch.equal(detector[..., 2], detector.new_full(detector[..., 2].shape, -2.0))
    source = operator.source_positions
    axis = source.new_tensor([0.0, 0.0, 0.0])
    torch.testing.assert_close((source - axis).norm(dim=-1), source.new_full((5,), 8.0))
    torch.testing.assert_close(source[2], source.new_tensor([0.0, 0.0, 8.0]))
    assert source[0, 0] < 0 < source[-1, 0]
    assert source[-1, 2] < source[2, 2]
    center = detector[0].mean(dim=(0, 1))
    torch.testing.assert_close(center, center.new_tensor([0.0, 0.0, -2.0]))
    assert detector[0, -1, 0, 0] > detector[0, 0, 0, 0]
    assert detector[0, 0, -1, 1] > detector[0, 0, 0, 1]


def test_chest_wall_source_sits_on_the_outer_edge_of_column_zero():
    angles = torch.tensor([0.0], dtype=torch.float64)
    _, detector = arc_positions(angles, (2, 5), 0.4, 10.0, 8.0, source_y=-1.0)
    edge = detector[0, 0, 0, 1] - 0.4 / 2
    torch.testing.assert_close(edge, detector.new_tensor(-1.0))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adjoint_matches_forward(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(4)
    operator = projector(ray_chunk=5).to(device)
    volume = torch.randn(2, 1, 4, 5, 6, dtype=torch.float64, device=device)
    projections = torch.randn(2, 1, 5, 3, 4, dtype=torch.float64, device=device)
    projected = operator(volume)
    back = operator.adjoint(projections)
    torch.testing.assert_close((projected * projections).sum(), (volume * back).sum(), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(projector(ray_chunk=1).to(device)(volume), projected)


def test_rejects_axis_behind_the_detector():
    with pytest.raises(ValueError, match="rotation axis"):
        projector(source_to_axis=10.0, source_to_detector=10.0)


class Adjoint(torch.nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, projections):
        return self.operator.adjoint(projections)


def test_learnable_geometry_is_two_parameters():
    operator = projector(learnable_geometry=True)
    names = dict(operator.named_parameters())
    assert set(names) == {"angles", "distances"}
    assert all(isinstance(value, torch.nn.Parameter) for value in names.values())
    torch.testing.assert_close(operator.distances, operator.distances.new_tensor([10.0, 8.0, 0.0]))
    assert not list(projector().parameters())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_geometry_gradients_match_finite_differences(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(5)
    operator = projector(learnable_geometry=True, source_y=-0.4).to(device)
    volume = torch.rand(1, 1, 4, 5, 6, dtype=torch.float64, device=device)
    projections = torch.rand(1, 1, 5, 3, 4, dtype=torch.float64, device=device)
    geometry = (operator.angles.detach().clone().requires_grad_(), operator.distances.detach().clone().requires_grad_())

    def forward(angles, distances):
        return torch.func.functional_call(operator, {"angles": angles, "distances": distances}, (volume,))

    def adjoint(angles, distances):
        parameters = {"operator.angles": angles, "operator.distances": distances}
        return torch.func.functional_call(Adjoint(operator), parameters, (projections,))

    assert torch.autograd.gradcheck(forward, geometry, atol=1e-5)
    assert torch.autograd.gradcheck(adjoint, geometry, atol=1e-5)


def test_repeated_optimiser_steps_use_the_current_geometry():
    torch.manual_seed(6)
    operator = projector(learnable_geometry=True)
    volume = torch.rand(1, 1, 4, 5, 6, dtype=torch.float64)
    optimiser = torch.optim.SGD(operator.parameters(), lr=1e-3)
    for _ in range(3):
        optimiser.zero_grad()
        operator(volume).square().sum().backward()
        assert operator.angles.grad.abs().sum() > 0 and operator.distances.grad.abs().sum() > 0
        before = operator(volume).detach()
        optimiser.step()
        with torch.no_grad():
            after = operator(volume)
        assert not torch.allclose(before, after)
        torch.testing.assert_close(after, operator(volume).detach())
    rebuilt = projector(
        angles=operator.angles.detach(),
        source_to_detector=operator.source_to_detector,
        source_to_axis=operator.source_to_axis,
        source_y=operator.source_y,
    )
    torch.testing.assert_close(operator.source_positions.detach(), rebuilt.source_positions)
    torch.testing.assert_close(copy.deepcopy(operator)(volume), operator(volume))


def test_a_fixed_geometry_builds_its_positions_once(monkeypatch):
    operator = projector()
    calls = []
    original = operator._positions
    monkeypatch.setattr(operator, "_positions", lambda: calls.append(1) or original())
    operator(torch.rand(1, 1, 4, 5, 6, dtype=torch.float64))
    assert not calls
    operator.angles.requires_grad_(True)
    operator(torch.rand(1, 1, 4, 5, 6, dtype=torch.float64)).sum().backward()
    assert calls and operator.angles.grad is not None


def test_recovers_an_arc_radius_and_angle_offset_from_projections():
    def arc(**kwargs):
        defaults = dict(
            angles=torch.linspace(-0.44, 0.44, 7, dtype=torch.float64),
            detector_shape=(16, 12),
            pixel_size=0.5,
            source_to_detector=20.0,
            source_to_axis=16.0,
            source_y=-3.0,
            voxel_spacing=0.5,
            volume_center=(0.0, 0.0, -2.0),
            n_samples=24,
            ray_chunk=4096,
            dtype=torch.float64,
        )
        defaults.update(kwargs)
        return BreastTomosynthesis((6, 8, 10), **defaults)

    torch.manual_seed(7)
    volume = torch.zeros(1, 1, 6, 8, 10, dtype=torch.float64)
    volume[..., 1:5, 2:6, 2:8] = torch.rand(1, 1, 4, 4, 6, dtype=torch.float64) + 0.5
    truth = arc(source_to_axis=15.0)
    measured = truth(volume)
    operator = arc(learnable_geometry=True)
    with torch.no_grad():
        operator.angles += 0.05
    optimiser = torch.optim.Adam(operator.parameters(), lr=1e-2)
    for _ in range(200):
        optimiser.zero_grad()
        (operator(volume) - measured).square().sum().backward()
        operator.distances.grad[[0, 2]] = 0
        optimiser.step()
    torch.testing.assert_close(operator.angles.detach(), truth.angles, atol=5e-3, rtol=0)
    assert operator.source_to_axis == pytest.approx(15.0, abs=0.05)
