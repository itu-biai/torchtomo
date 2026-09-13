"""Seeded ellipse phantoms, transmission Poisson noise, and FBP calibration."""

import logging
import math

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger("ellipses")


def make_phantoms(size=64, count=100, seed=2026):
    """Draw independent random ellipses on a 3x grid, then average to pixels."""
    generator = torch.Generator().manual_seed(seed)
    coords = torch.linspace(-1, 1, size * 3)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    support = x.square() + y.square() < 0.93**2

    def uniform(low, high):
        return low + (high - low) * torch.rand((), generator=generator).item()

    phantoms, parameters = [], []
    for _ in range(count):
        ellipses = []
        image = torch.zeros_like(x)
        # A low-intensity body plus independently positioned bright/dark objects.
        ellipses.append([uniform(0.12, 0.3), uniform(0.65, 0.85), uniform(0.65, 0.85), 0, 0, uniform(0, math.pi)])
        for _ in range(int(torch.randint(6, 13, (), generator=generator))):
            intensity = uniform(0.15, 0.85)
            if uniform(0, 1) < 0.2:
                intensity *= -0.5
            ellipses.append(
                [
                    intensity,
                    uniform(0.06, 0.35),
                    uniform(0.06, 0.28),
                    uniform(-0.55, 0.55),
                    uniform(-0.55, 0.55),
                    uniform(0, math.pi),
                ]
            )
        for intensity, a, b, cx, cy, angle in ellipses:
            xr = math.cos(angle) * (x - cx) + math.sin(angle) * (y - cy)
            yr = -math.sin(angle) * (x - cx) + math.cos(angle) * (y - cy)
            image += intensity * ((xr / a).square() + (yr / b).square() <= 1)
        image = image.clamp_min(0) * support
        image /= image.max().clamp_min(1e-8)
        phantoms.append(F.avg_pool2d(image[None, None], 3)[0])
        parameters.append(ellipses)
    return torch.stack(phantoms), parameters


def make_splits(seed=2027):
    permutation = torch.randperm(100, generator=torch.Generator().manual_seed(seed))
    return {"train": permutation[:60], "val": permutation[60:80], "test": permutation[80:]}


def poisson_sinogram(clean, photons, seed):
    """N ~ Poisson(I0 exp(-Ax)); y = -log(max(N, 1) / I0).

    Sampling is on CPU for consistent random streams across training devices.
    Negative post-log samples are retained. Only zero counts are floored to one.
    """
    generator = torch.Generator().manual_seed(seed)
    counts = torch.poisson(photons * torch.exp(-clean), generator=generator)
    noisy = -torch.log(counts.clamp_min(1) / photons)
    return noisy, counts


def psnr_per_image(prediction, target):
    """Full-image PSNR, fixed data range 1, with no prediction clipping."""
    mse = (prediction - target).square().flatten(1).mean(1)
    return -10 * torch.log10(mse.clamp_min(1e-12))


@torch.no_grad()
def apply_in_batches(operation, tensor, batch_size=5):
    return torch.cat([operation(batch) for batch in tensor.split(batch_size)])


def calibrate_photons(projector, clean_train, truth_train, target=23.0, seed=2028):
    """Choose I0 on training images only; never inspect validation/test targets."""
    trials = []

    def evaluate(photons):
        noisy, _ = poisson_sinogram(clean_train, photons, seed)
        reconstruction = apply_in_batches(projector.fbp, noisy)
        score = psnr_per_image(reconstruction, truth_train).mean().item()
        trials.append({"photons": photons, "train_fbp_psnr_db": score})
        LOGGER.info("calibration photons=%.3f train_fbp_psnr_db=%.4f", photons, score)
        return score

    low, high = 10.0, 1e6
    low_score, high_score = evaluate(low), evaluate(high)
    if not low_score <= target <= high_score:
        raise ValueError(f"Target {target:.2f} dB is not bracketed by {low_score:.2f} and {high_score:.2f} dB")
    for _ in range(12):
        midpoint = math.sqrt(low * high)
        score = evaluate(midpoint)
        if abs(score - target) < 0.05:
            break
        if score < target:
            low = midpoint
        else:
            high = midpoint
    best = min(trials, key=lambda trial: abs(trial["train_fbp_psnr_db"] - target))
    return best["photons"], trials
