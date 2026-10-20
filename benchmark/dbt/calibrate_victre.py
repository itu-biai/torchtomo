"""Self-calibration of the VICTRE arc from its own projections, alternating volume and geometry updates.

Each case starts from a geometry, fits the volume with SIRT, and then alternates one Adam step on the
chosen geometry parameters, against the masked projection residual with the volume fixed, with a few warm
started SIRT iterations. Agreement with VICTRE's FBP is measured only before and after, never optimised.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torchtomo.victre import N_VIEWS, SOURCE_TO_DETECTOR_MM, victre_projector

BINS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKEND = "auto"


def load(path: Path):
    coarsen = BINS // 4
    data = np.load(path)
    projections = torch.as_tensor(data["projections"])[None]
    valid = torch.as_tensor(data["valid"]).float()[None]
    weight = F.avg_pool2d(valid, coarsen)
    measured = F.avg_pool2d(projections * valid, coarsen) / weight.clamp_min(1e-6)
    reference = F.avg_pool2d(torch.as_tensor(data["reference"]), coarsen).transpose(-1, -2)
    return (
        measured[None].to(DEVICE),
        (weight > 0.999)[None].to(DEVICE),
        reference[None, None].contiguous().to(DEVICE),
    )


def projector(margin=4, **overrides):
    depth = 57 + 2 * margin
    return victre_projector(
        "scattered",
        bin_factor=BINS,
        depth_margin=margin,
        n_samples=math.ceil(1.3 * depth / (0.085 * BINS)),
        ray_chunk=32768,
        backend=BACKEND,
        device=DEVICE,
        **overrides,
    )


def sirt(operator, measured, valid, volume, iterations):
    with torch.no_grad():
        row = operator(torch.ones_like(volume))
        column = operator.adjoint(valid.float())
        row_weight = torch.where(row > 1e-6, 1 / row, 0) * valid
        column_weight = torch.where(column > 1e-6, 1 / column, 0)
        for _ in range(iterations):
            volume = (volume + operator.adjoint((measured - operator(volume)) * row_weight) * column_weight).clamp_min(
                0
            )
    return volume


def hann_ramp_filter(projections, operator):
    detector = operator.detector_positions[0].detach()
    from_source = detector[..., 1] - operator.source_positions[0, 1].detach()
    weights = SOURCE_TO_DETECTOR_MM / (SOURCE_TO_DETECTOR_MM**2 + detector[..., 0] ** 2 + from_source**2).sqrt()
    weighted = projections * weights
    rows = weighted.shape[-2]
    size = 2 ** math.ceil(math.log2(rows * 1.6))
    low = (size - rows) // 2
    padded = F.pad(weighted.flatten(0, 2).transpose(-1, -2), (low, size - rows - low), mode="replicate")
    frequency = torch.fft.rfftfreq(size, device=projections.device)
    ramp = 2 * frequency * (0.5 + 0.5 * torch.cos(math.pi * frequency / frequency.max()))
    filtered = torch.fft.irfft(torch.fft.rfft(padded, dim=-1) * ramp, n=size, dim=-1)[..., low : low + rows]
    return filtered.transpose(-1, -2).reshape(weighted.shape)


def correlation(left, right, mask):
    x, y = left[mask].double(), right[mask].double()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()))


def fbp_correlation(geometry, measured, valid, reference, mask):
    operator = projector(margin=0, **geometry)
    with torch.no_grad():
        return correlation(operator.adjoint(hann_ramp_filter(measured * valid, operator)), reference, mask)


def geometry_of(operator):
    return {
        "angles": operator.angles.detach().clone(),
        "source_to_detector": operator.source_to_detector,
        "source_to_axis": operator.source_to_axis,
        "source_y": operator.source_y,
    }


def calibrate(name, start, learn, measured, valid, reference, mask, outer, inner):
    operator = projector(learnable_geometry=True, **start)
    shape = operator.operator.volume_shape
    volume = sirt(operator, measured, valid, torch.zeros(1, 1, *shape, device=DEVICE), 10)
    angle_lr, distance_lr = 2e-3, 2.0
    optimiser = torch.optim.Adam(
        [{"params": [operator.angles], "lr": angle_lr}, {"params": [operator.distances], "lr": distance_lr}]
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=outer, eta_min=0.0)
    distance_mask = torch.tensor([0.0, "radius" in learn, "source_y" in learn], device=DEVICE)
    published = torch.linspace(-math.radians(25), math.radians(25), N_VIEWS, device=DEVICE)
    history = []
    started = time.perf_counter()
    for step in range(outer + 1):
        residual = valid * (operator(volume) - measured)
        loss = residual.square().sum()
        relative = float(residual.detach().norm() / (valid * measured).norm())
        angle_error = (operator.angles.detach() - published) * 180 / math.pi
        history.append(
            {
                "step": step,
                "relative_residual": relative,
                "mean_angle_error_deg": float(angle_error.mean()),
                "max_angle_error_deg": float(angle_error.abs().max()),
                "source_to_axis": operator.source_to_axis,
                "source_y": operator.source_y,
            }
        )
        if step % 10 == 0:
            print(name, json.dumps(history[-1]), f"{time.perf_counter() - started:.0f}s", flush=True)
        if step == outer:
            break
        optimiser.zero_grad()
        loss.backward()
        if "angles" not in learn:
            operator.angles.grad.zero_()
        operator.distances.grad *= distance_mask
        optimiser.step()
        schedule.step()
        volume = sirt(operator, measured, valid, volume, inner)
    return {
        "start": {key: (value.tolist() if torch.is_tensor(value) else value) for key, value in start.items()},
        "learn": learn,
        "fbp_correlation_start": fbp_correlation(start, measured, valid, reference, mask),
        "fbp_correlation_calibrated": fbp_correlation(geometry_of(operator), measured, valid, reference, mask),
        "seconds": time.perf_counter() - started,
        "history": history,
    }


def main(args):
    global BINS, BACKEND
    BINS, BACKEND = args.bins, args.backend
    measured, valid, reference = load(args.input)
    published = projector(margin=0)
    with torch.no_grad():
        outside = published.adjoint(((measured <= 0) & valid).float())
        coverage = published.adjoint(valid.float())
    hull = (outside <= 1e-3 * coverage.max()) & (coverage > 0.5 * coverage.max())
    mask = hull & (reference > 0.08 * torch.quantile(reference[hull], 0.995))
    angles = torch.linspace(-math.radians(25), math.radians(25), N_VIEWS, device=DEVICE)
    cases = {
        "published": ({}, ["angles", "radius", "source_y"]),
        "angles_plus_3deg": ({"angles": angles + math.radians(3)}, ["angles"]),
        "source_over_centre": ({"source_y": 0.0}, ["source_y"]),
        "radius_630": ({"source_to_axis": 630.0}, ["radius"]),
    }
    selected = args.cases or list(cases)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {
        "bins": BINS,
        "device": str(DEVICE),
        "fbp_correlation_published": fbp_correlation({}, measured, valid, reference, mask),
    }
    print("published FBP correlation", results["fbp_correlation_published"], flush=True)
    for name in selected:
        start, learn = cases[name]
        results[name] = calibrate(name, start, learn, measured, valid, reference, mask, args.outer, args.inner)
        summary = {key: value for key, value in results[name].items() if key != "history"}
        summary["end"] = results[name]["history"][-1]
        print(name, "SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "start"}), flush=True)
        (args.output / "results.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input_bin4.npz from prepare_victre.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=16, help="detector binning in native pixels, a multiple of 4")
    parser.add_argument("--outer", type=int, default=50, help="geometry steps")
    parser.add_argument("--inner", type=int, default=2, help="SIRT iterations between geometry steps")
    parser.add_argument("--backend", choices=("auto", "torch", "cuda"), default="auto")
    parser.add_argument("cases", nargs="*")
    main(parser.parse_args())
