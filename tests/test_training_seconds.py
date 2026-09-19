"""Wall-clock seconds must land on every scored method, including classical."""

import argparse
import sys
from pathlib import Path

import torch

from torchtomo import FanBeam, ParallelBeam

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "ellipses"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))


def _classical_args(**overrides):
    values = dict(
        device="cpu",
        batch_size=2,
        sirt_iterations=2,
        sirt_relaxation=1.0,
        sart_sweeps=1,
        sart_subsets=2,
        sart_relaxations="1",
        red_iterations=1,
        red_step=1.0,
        red_weights="1",
        bm3d_sigmas="0.1",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_classical_records_positive_seconds():
    from classical import run_classical

    projector = FanBeam(img_size=16, n_angles=8, n_samples=8)
    n = 4
    data = {
        "noisy": torch.randn(n, 1, projector.n_angles, projector.n_det),
        "truth": torch.rand(n, 1, 16, 16),
        "fbp": torch.rand(n, 1, 16, 16),
        "splits": {"val": torch.tensor([0, 1]), "test": torch.tensor([2, 3])},
    }
    reconstructions, selection = run_classical(
        projector, data, None, 1.0, _classical_args(), projector.circle_mask.bool()
    )
    assert set(selection) >= {"sirt", "sart", "bm3d"}
    assert "red" not in selection
    for name, record in selection.items():
        assert record["seconds"] > 0
        assert reconstructions[name].shape == (2, 1, 16, 16)


def test_record_method_seconds_covers_classical_and_learned():
    from classical import run_classical
    from train import fbp_split_seconds, record_method_seconds

    projector = ParallelBeam(img_size=16, n_angles=8)
    n = 4
    noisy = torch.randn(n, 1, projector.n_angles, projector.n_det)
    data = {
        "noisy": noisy,
        "truth": torch.rand(n, 1, 16, 16),
        "fbp": torch.rand(n, 1, 16, 16),
        "splits": {"val": torch.tensor([0, 1]), "test": torch.tensor([2, 3])},
    }
    _, selection = run_classical(projector, data, None, 1.0, _classical_args(), projector.circle_mask.bool())
    _, fbp_seconds = fbp_split_seconds(projector, noisy[2:], "cpu", 2)
    training = {"lpd": {"training_seconds": 1.25}, "fbp-unet": {"training_seconds": 0.5}}
    metrics = {name: {} for name in ("fbp", "sirt", "sart", "bm3d", "fbp-unet", "lpd")}
    record_method_seconds(metrics, training, selection, fbp_seconds)
    for name, row in metrics.items():
        assert row["seconds"] > 0, name
    assert metrics["lpd"]["seconds"] == training["lpd"]["training_seconds"]
    assert metrics["sirt"]["seconds"] == selection["sirt"]["seconds"]
    assert metrics["fbp"]["seconds"] == fbp_seconds
