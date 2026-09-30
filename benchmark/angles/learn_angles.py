"""Which K views to measure: angles learnt through the pose table's angle column.

Run from the repository root:
    PYTHONPATH=src python benchmark/angles/learn_angles.py

A sparse-view scan measures K projections. Which K is a design question whose
answer depends on what is being scanned: a class of objects whose edges share
an orientation, such as fibres or laminates, is seen best from the angles that
run along those edges. Here the angles are learnt by gradient descent on the
reconstruction error over a training set, differentiating the measurement
(forward) and every iteration of the reconstruction (forward and adjoint) with
respect to the angle column.

The reconstruction is projected Landweber (SIRT without row and column
weights), unrolled for a fixed number of iterations, so that it treats any set
of angles fairly: FBP's quadrature assumes evenly spaced views.

Two object classes, each with its own test set:

- "fibres": thin ellipses, their long axes within about 15 degrees of one
  direction. Learnt angles should beat evenly spaced ones.
- "isotropic": the same ellipses at any orientation. This is the control: evenly
  spaced views are hard to beat, and a learnt set that did would be suspect.

Baselines: evenly spaced, and random sets drawn uniformly (the mean over draws).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from torchtomo import ParallelBeam

RESULTS = Path(__file__).resolve().parent / "results"


def ellipses(count, size, orientation_spread_degrees, generator, device):
    """Random phantoms of 4 to 10 thin ellipses each, [count, 1, size, size] in [0, 1]."""
    coordinates = torch.linspace(-1, 1, size, device=device)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    images = torch.zeros(count, 1, size, size, device=device)
    for index in range(count):
        for _ in range(int(torch.randint(4, 11, (1,), generator=generator))):
            u = torch.rand(6, generator=generator).tolist()
            centre_x, centre_y = 1.1 * (u[0] - 0.5), 1.1 * (u[1] - 0.5)
            length = 0.15 + 0.35 * u[2]
            width = length / (3 + 5 * u[3])
            if orientation_spread_degrees is None:
                angle = math.pi * u[4]
            else:
                angle = math.radians(orientation_spread_degrees) * float(torch.randn((), generator=generator))
            value = 0.3 + 0.7 * u[5]
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            along = (x - centre_x) * cos_a + (y - centre_y) * sin_a
            across = -(x - centre_x) * sin_a + (y - centre_y) * cos_a
            inside = (along / length) ** 2 + (across / width) ** 2 <= 1
            images[index, 0] += value * inside
    images = images.clamp(max=1.0)
    return images * (x**2 + y**2 <= 0.95**2)


def set_angles(projector, angles):
    projector.pose = torch.stack([angles, torch.zeros_like(angles)], dim=1)


@torch.no_grad()
def step_size(projector, size, device, iterations=20):
    """1 / ||A||^2 by power iteration, for the current angles."""
    image = torch.rand(1, 1, size, size, device=device)
    for _ in range(iterations):
        image = projector.adjoint(projector.forward(image))
        norm = image.norm()
        image = image / norm
    return 1.0 / norm.item()


def reconstruct(projector, sinogram, size, iterations, step):
    image = torch.zeros(sinogram.shape[0], 1, size, size, device=sinogram.device, dtype=sinogram.dtype)
    for _ in range(iterations):
        image = (image - step * projector.adjoint(projector.forward(image) - sinogram)).clamp(min=0)
    return image


def psnr(images, truth):
    mse = (images - truth).pow(2).flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def evaluate(projector, angles, test, size, iterations, noise, generator):
    with torch.no_grad():
        set_angles(projector, angles)
        step = step_size(projector, size, test.device)
        sinogram = projector.forward(test)
        sinogram = sinogram + noise * sinogram.std() * torch.randn(sinogram.shape, generator=generator).to(test.device)
        return psnr(reconstruct(projector, sinogram, size, iterations, step), test)


def learn(projector, initial, size, iterations, noise, spread, steps, batch, lr, generator, device):
    angles = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([angles], lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr / 20)
    for _ in range(steps):
        truth = ellipses(batch, size, spread, generator, device)
        with torch.no_grad():
            set_angles(projector, angles.detach())
            step = step_size(projector, size, device)
        optimizer.zero_grad()
        set_angles(projector, angles)
        sinogram = projector.forward(truth)
        noise_sample = torch.randn(sinogram.shape, generator=generator).to(device)
        sinogram = sinogram + noise * sinogram.detach().std() * noise_sample
        loss = (reconstruct(projector, sinogram, size, iterations, step) - truth).pow(2).mean()
        loss.backward()
        optimizer.step()
        schedule.step()
    return angles.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=30, help="unrolled Landweber iterations")
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--spread", type=float, default=15.0, help="fibre orientation std in degrees")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01, help="radians per step")
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--random-draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    RESULTS.mkdir(exist_ok=True)

    projector = ParallelBeam(img_size=args.size, n_angles=args.views, circle=True, backend="auto").to(device)
    uniform = projector.pose[:, 0].detach().clone()
    report = {}
    for label, spread in (("fibres", args.spread), ("isotropic", None)):
        generator = torch.Generator().manual_seed(args.seed)
        test = ellipses(args.test, args.size, spread, torch.Generator().manual_seed(10_000 + args.seed), device)
        start = time.perf_counter()
        learnt = learn(
            projector,
            uniform,
            args.size,
            args.iterations,
            args.noise,
            spread,
            args.steps,
            args.batch,
            args.lr,
            generator,
            device,
        )
        seconds = time.perf_counter() - start
        scores = {}
        for name, angles in (("uniform", uniform), ("learnt", learnt)):
            scores[name] = evaluate(
                projector, angles, test, args.size, args.iterations, args.noise, torch.Generator().manual_seed(1)
            )
        draws = []
        for draw in range(args.random_draws):
            angles = math.pi * torch.rand(args.views, generator=torch.Generator().manual_seed(100 + draw))
            draws.append(
                evaluate(
                    projector,
                    angles.to(device, uniform.dtype),
                    test,
                    args.size,
                    args.iterations,
                    args.noise,
                    torch.Generator().manual_seed(1),
                )
            )
        scores["random"] = float(np.mean(draws))
        folded = np.sort(np.rad2deg(learnt.cpu().numpy()) % 180)
        report[label] = dict(psnr=scores, learnt_degrees=folded.tolist(), train_seconds=seconds)
        print(
            f"{label:9s} uniform {scores['uniform']:.2f} dB  random {scores['random']:.2f} dB  "
            f"learnt {scores['learnt']:.2f} dB  ({seconds:.0f} s)",
            flush=True,
        )
        print(f"          learnt angles (deg, mod 180): {[round(float(v), 1) for v in folded]}", flush=True)
    (RESULTS / f"angles_{args.views}.json").write_text(json.dumps(dict(args=vars(args), report=report), indent=1))


if __name__ == "__main__":
    main()
