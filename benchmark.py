#!/usr/bin/env python
"""Benchmark torchtomo forward and back projection throughput."""

import time

import numpy as np
import torch
from skimage.transform import iradon, radon

from torchtomo import FanBeam, ParallelBeam, shepp_logan


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


def bench(fn, n_warmup=3, n_runs=20):
    for _ in range(n_warmup):
        fn()
    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    elapsed = time.perf_counter() - start
    return n_runs / elapsed


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


def benchmark_batch_forward(
    projector, phantom, batch_size, device, n_warmup=3, n_runs=20
):
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


def benchmark_skimage(img_sizes, n_angles=180):
    print(f"\n{'=' * 60}")
    print("scikit-image (CPU only)")
    print("=" * 60)

    for img_size in img_sizes:
        print(f"\n--- Image size: {img_size}x{img_size} ---")

        phantom_np = np.random.RandomState(42).rand(img_size, img_size).astype(
            np.float32
        )
        theta = np.linspace(0, 180, n_angles, endpoint=False)

        fwd_rate = bench(lambda: radon(phantom_np, theta=theta))
        sino_sk = radon(phantom_np, theta=theta)
        fbp_rate = bench(lambda: iradon(sino_sk, theta=theta, filter_name="ramp"))

        print(f"\nParallel Beam (single slice):")
        print(f"  Forward:  {fwd_rate:>8.1f} slices/sec")
        print(f"  FBP:      {fbp_rate:>8.1f} slices/sec")

    return fwd_rate, fbp_rate


def benchmark_torchtomo(img_sizes, batch_sizes, devices):
    for device in devices:
        print(f"\n{'=' * 60}")
        print(f"torchtomo — Device: {device}")
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
                    rate = benchmark_batch_forward(
                        projector, phantom, batch_size, device
                    )
                    print(f"  Batch {batch_size:>2}: {rate:>8.1f} slices/sec")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  Batch {batch_size:>2}: OOM")
                        break
                    raise


def benchmark_comparison(img_sizes, n_angles=180):
    print(f"\n{'=' * 60}")
    print("Head-to-Head: torchtomo vs scikit-image (Parallel Beam)")
    print("=" * 60)

    devices = get_available_devices()

    for img_size in img_sizes:
        print(f"\n--- {img_size}x{img_size}, {n_angles} angles ---")

        phantom_np = np.random.RandomState(42).rand(img_size, img_size).astype(
            np.float32
        )
        phantom_t = torch.from_numpy(phantom_np).unsqueeze(0).unsqueeze(0)
        theta = np.linspace(0, 180, n_angles, endpoint=False)

        sk_fwd = bench(lambda: radon(phantom_np, theta=theta))
        sino_sk = radon(phantom_np, theta=theta)
        sk_fbp = bench(lambda: iradon(sino_sk, theta=theta, filter_name="ramp"))

        results = [("scikit-image", sk_fwd, sk_fbp)]

        for device in devices:
            label = f"torchtomo ({device})"
            proj = ParallelBeam(img_size=img_size, n_angles=n_angles, n_det=img_size)
            proj = proj.to(device)
            pt = phantom_t.to(device)
            sino_t = proj.forward(pt)
            sync_device(device)

            def fwd(p=proj, x=pt, d=device):
                p.forward(x)
                sync_device(d)

            def fbp(p=proj, s=sino_t, d=device):
                p.fbp(s)
                sync_device(d)

            tt_fwd = bench(fwd)
            tt_fbp = bench(fbp)
            results.append((label, tt_fwd, tt_fbp))

        print(f"  {'':>22} | {'Forward (sl/s)':>15} | {'FBP (sl/s)':>15}")
        print(f"  {'-' * 22}-+-{'-' * 15}-+-{'-' * 15}")
        for label, fwd_rate, fbp_rate in results:
            print(f"  {label:>22} | {fwd_rate:>15.1f} | {fbp_rate:>15.1f}")

        print()
        print(f"  Speedup vs scikit-image:")
        for label, fwd_rate, fbp_rate in results[1:]:
            print(
                f"    {label}: forward {fwd_rate / sk_fwd:.1f}x, FBP {fbp_rate / sk_fbp:.1f}x"
            )


def main():
    print("=" * 60)
    print("TorchTomo Benchmark")
    print("=" * 60)

    devices = get_available_devices()
    print(f"\nAvailable devices: {[str(d) for d in devices]}")

    img_sizes = [256, 512]
    batch_sizes = [1, 4, 8, 16]

    benchmark_skimage(img_sizes)
    benchmark_torchtomo(img_sizes, batch_sizes, devices)
    benchmark_comparison(img_sizes)

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
