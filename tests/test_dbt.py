"""Arc geometry, stationary detector, and matched transpose for breast tomosynthesis."""

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
