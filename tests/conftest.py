"""Shared pytest fixtures for torchtomo tests."""

import pytest
import torch

from torchtomo import FanBeam, ParallelBeam, shepp_logan


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def img_size():
    return 128


@pytest.fixture
def phantom_small(img_size, device):
    return shepp_logan(img_size, device=device)


@pytest.fixture
def parallel_projector(img_size):
    return ParallelBeam(img_size=img_size, n_angles=90, n_det=img_size)


@pytest.fixture
def fan_projector(img_size):
    return FanBeam(
        img_size=img_size,
        n_angles=180,
        n_det=200,
        src_dist=500,
        det_dist=500,
    )
