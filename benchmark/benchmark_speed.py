#!/usr/bin/env python
"""Time torchtomo's forward, adjoint, and FBP on every device and backend available.

Each row builds one projector, then times its three operators on the same batch
in milliseconds per call. On CUDA the card is kept busy for a moment first, so
the first rows do not pay for its clocks ramping up.

    PYTHONPATH=src python benchmark/benchmark_speed.py
    PYTHONPATH=src python benchmark/benchmark_speed.py --devices cuda --sizes 512 --angles 360 90

Comparisons against other projectors (LEAP, torch-radon, scikit-image) live in
https://github.com/itu-biai/torchtomo-benchmark.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from torchtomo import FanBeam, ParallelBeam
from torchtomo._cuda_kernels import cuda_kernels_available
from torchtomo._triton_kernels import triton_kernels_available


def get_available_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def warm(device, seconds=1.0):
    """Spin the card so its clocks are up before the first timed row."""
    if device.type != "cuda":
        return
    a = torch.rand(2048, 2048, device=device)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        a @ a
    torch.cuda.synchronize()


def milliseconds(fn, device, warmup=3, budget=0.5, max_runs=50):
    """Mean milliseconds per call, running until the time budget or max_runs is spent."""
    for _ in range(warmup):
        fn()
    sync_device(device)
    runs = 0
    start = time.perf_counter()
    while runs < max_runs and (runs < 3 or time.perf_counter() - start < budget):
        fn()
        runs += 1
        sync_device(device)
    return (time.perf_counter() - start) * 1000 / runs


def backends(geometry, device):
    """(label, constructor options) for each backend that runs natively on this device."""
    rows = [("torch", {})]
    if geometry == "parallel" and triton_kernels_available(device, torch.float32):
        rows.append(("triton", {"backend": "triton"}))
    if cuda_kernels_available(device, torch.float32):
        rows.append(("cuda", {"backend": "cuda"}))
        rows.append(("cuda, approximate", {"backend": "cuda", "approximate": True}))
    return rows


def build(geometry, size, n_angles, options, device):
    projector_class = ParallelBeam if geometry == "parallel" else FanBeam
    return projector_class(img_size=size, n_angles=n_angles, **options).to(device)


def held_megabytes(device, before, sinogram):
    """Memory the projector keeps between calls: its tables and any cached grids."""
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    return (torch.cuda.memory_allocated() - before - sinogram.nbytes) / 2**20


@torch.no_grad()
def time_operators(projector, image, sinogram, device):
    return dict(
        forward_ms=milliseconds(lambda: projector.forward(image), device),
        adjoint_ms=milliseconds(lambda: projector.adjoint(sinogram), device),
        fbp_ms=milliseconds(lambda: projector.fbp(sinogram), device),
    )


def run(args):
    rows = []
    for device in args.devices:
        warm(device)
        for geometry in args.geometries:
            for size in args.sizes:
                for n_angles in args.angles:
                    image = torch.rand(args.batch_size, 1, size, size, device=device)
                    for label, options in backends(geometry, device):
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        before = torch.cuda.memory_allocated() if device.type == "cuda" else 0
                        projector = build(geometry, size, n_angles, options, device)
                        sinogram = torch.rand(args.batch_size, 1, n_angles, projector.n_det, device=device)
                        row = dict(
                            device=device.type,
                            geometry=geometry,
                            size=size,
                            angles=n_angles,
                            batch=args.batch_size,
                            backend=label,
                            **time_operators(projector, image, sinogram, device),
                        )
                        row["held_mb"] = held_megabytes(device, before, sinogram)
                        rows.append(row)
                        print(format_row(row), flush=True)
                        del projector, sinogram
    return rows


def format_row(row):
    held = "" if row["held_mb"] is None else f"{row['held_mb']:.0f}"
    return (
        f"| {row['device']} | {row['geometry']} | {row['size']} | {row['angles']} | {row['backend']} "
        f"| {row['forward_ms']:.2f} | {row['adjoint_ms']:.2f} | {row['fbp_ms']:.2f} | {held} |"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", nargs="+", type=torch.device, default=get_available_devices())
    parser.add_argument("--geometries", nargs="+", choices=("parallel", "fan"), default=["parallel", "fan"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--angles", nargs="+", type=int, default=[360, 90])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--json", type=Path, help="also write the rows to this file")
    args = parser.parse_args()

    print("| device | geometry | size | angles | backend | forward ms | adjoint ms | FBP ms | held MB |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = run(args)
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
