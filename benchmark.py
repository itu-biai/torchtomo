#!/usr/bin/env python
"""Benchmark torchtomo forward and back projection throughput."""

import time
import torch

from torchtomo import ParallelBeam, FanBeam, shepp_logan


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


def benchmark_forward(projector, phantom, device, n_warmup=5, n_runs=50):
    for _ in range(n_warmup):
        _ = projector.forward(phantom)

    sync_device(device)

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = projector.forward(phantom)

    sync_device(device)

    elapsed = time.perf_counter() - start
    return n_runs / elapsed


def benchmark_fbp(projector, sinogram, device, n_warmup=5, n_runs=50):
    for _ in range(n_warmup):
        _ = projector.fbp(sinogram)

    sync_device(device)

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = projector.fbp(sinogram)

    sync_device(device)

    elapsed = time.perf_counter() - start
    return n_runs / elapsed


def benchmark_batch_forward(projector, phantom, batch_size, device, n_warmup=3, n_runs=20):
    batch = phantom.expand(batch_size, -1, -1, -1).clone()

    for _ in range(n_warmup):
        _ = projector.forward(batch)

    sync_device(device)

    start = time.perf_counter()
    for _ in range(n_runs):
        _ = projector.forward(batch)

    sync_device(device)

    elapsed = time.perf_counter() - start
    return (n_runs * batch_size) / elapsed


def main():
    print("=" * 60)
    print("TorchTomo Benchmark")
    print("=" * 60)

    devices = get_available_devices()
    print(f"\nAvailable devices: {[str(d) for d in devices]}")

    img_sizes = [256, 512]
    batch_sizes = [1, 4, 8, 16]

    for device in devices:
        print(f"\n{'=' * 60}")
        print(f"Device: {device}")
        print("=" * 60)

        for img_size in img_sizes:
            print(f"\n--- Image size: {img_size}x{img_size} ---")

            phantom = shepp_logan(img_size).to(device)

            # Parallel Beam
            projector = ParallelBeam(
                img_size=img_size,
                n_angles=180,
                n_det=img_size,
            ).to(device)
            sinogram = projector.forward(phantom)

            fwd_rate = benchmark_forward(projector, phantom, device)
            fbp_rate = benchmark_fbp(projector, sinogram, device)

            print(f"\nParallel Beam (single slice):")
            print(f"  Forward:  {fwd_rate:>8.1f} slices/sec")
            print(f"  FBP:      {fbp_rate:>8.1f} slices/sec")

            # Fan Beam
            projector = FanBeam(
                img_size=img_size,
                n_angles=360,
                n_det=int(img_size * 1.5),
                src_dist=img_size * 2,
                det_dist=img_size * 2,
            ).to(device)
            sinogram = projector.forward(phantom)

            fwd_rate = benchmark_forward(projector, phantom, device)
            fbp_rate = benchmark_fbp(projector, sinogram, device)

            print(f"\nFan Beam (single slice):")
            print(f"  Forward:  {fwd_rate:>8.1f} slices/sec")
            print(f"  FBP:      {fbp_rate:>8.1f} slices/sec")

            # Batch benchmarks
            print(f"\nBatch Forward Projection (Parallel Beam):")
            projector = ParallelBeam(
                img_size=img_size,
                n_angles=180,
                n_det=img_size,
            ).to(device)

            for batch_size in batch_sizes:
                try:
                    rate = benchmark_batch_forward(projector, phantom, batch_size, device)
                    print(f"  Batch {batch_size:>2}: {rate:>8.1f} slices/sec")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  Batch {batch_size:>2}: OOM")
                        break
                    raise

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
