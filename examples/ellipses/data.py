"""Ellipse phantoms, real CT slices, transmission Poisson noise, and FBP calibration."""

import logging
import math
from pathlib import Path

import numpy as np
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


def psnr_per_image(prediction, target, region=None):
    """PSNR per image, fixed data range 1, with no prediction clipping.

    Without a region this averages over the whole square, which includes the
    corners every method sets to zero. Pass a boolean [H, W] region to score
    only the part of the image the geometry can actually see.
    """
    error = (prediction - target).square()
    if region is None:
        mse = error.flatten(1).mean(1)
    else:
        # The projector may live on an accelerator while predictions are gathered on CPU.
        mse = error[:, 0][:, region.to(error.device)].mean(1)
    return -10 * torch.log10(mse.clamp_min(1e-12))


def load_ct_splits(data_dir, image_size):
    """Read packed CT slices written by pack_ct_subset.py.

    Returns the stacked ground truth, index tensors for each split, and the
    source file names. The packed splits are patient-disjoint by construction.
    """
    images, names, windows, splits, start = [], [], [], {}, 0
    for split in ("train", "val", "test"):
        path = Path(data_dir) / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run pack_ct_subset.py first")
        with np.load(path, allow_pickle=False) as archive:
            batch = torch.from_numpy(archive["images"].astype(np.float32)).unsqueeze(1)
            names.extend(str(name) for name in archive["names"])
            if "windows" in archive:
                windows.append(torch.from_numpy(archive["windows"].astype(np.float32)))
        if batch.shape[-1] != image_size:
            batch = F.interpolate(batch, size=(image_size, image_size), mode="area")
        images.append(batch)
        splits[split] = torch.arange(start, start + len(batch))
        start += len(batch)
    stacked = torch.cat(windows) if len(windows) == 3 else None
    return torch.cat(images), splits, names, stacked


def apply_window(images, windows):
    """Map attenuation into each slice's own display window, clipped to [0, 1].

    This reproduces the clipped image the dataset ships, so metrics and figures
    read at the contrast the window was chosen for instead of across the whole
    attenuation range, where soft tissue occupies a small part of the scale.
    """
    low = windows[:, 0].view(-1, 1, 1, 1)
    high = windows[:, 1].view(-1, 1, 1, 1)
    return ((images - low) / (high - low)).clamp(0, 1)


@torch.no_grad()
def apply_in_batches(operation, tensor, batch_size=5):
    return torch.cat([operation(batch) for batch in tensor.split(batch_size)])


def calibrate_photons(projector, clean_train, truth_train, target=23.0, seed=2028, windows=None):
    """Choose I0 on training images only; never inspect validation/test targets.

    With a display window the target is scored inside it, because that is the
    contrast the reconstruction is read at. A dose that looks acceptable across the
    full attenuation range can be pure noise once a narrow window is applied.
    """
    trials = []

    def evaluate(photons):
        noisy, _ = poisson_sinogram(clean_train, photons, seed)
        reconstruction = apply_in_batches(projector.fbp, noisy)
        if windows is None:
            score = psnr_per_image(reconstruction, truth_train).mean().item()
        else:
            score = (
                psnr_per_image(apply_window(reconstruction, windows), apply_window(truth_train, windows)).mean().item()
            )
        trials.append({"photons": photons, "train_fbp_psnr_db": score})
        LOGGER.info("calibration photons=%.3f train_fbp_psnr_db=%.4f", photons, score)
        return score

    low, high = 10.0, 1e8
    low_score, high_score = evaluate(low), evaluate(high)
    if not low_score <= target <= high_score:
        raise ValueError(f"Target {target:.2f} dB is not bracketed by {low_score:.2f} and {high_score:.2f} dB")
    for _ in range(16):
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
