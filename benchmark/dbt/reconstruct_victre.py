"""FBP and OS-SART of one prepared VICTRE phantom, with two wrong-geometry controls.

FBP is the BIAI Lab DBT 2026v1 filter (cosine weighting from the source at 0
degrees, a Hann-windowed ramp along tube travel) followed by the matched
transpose of ``BreastTomosynthesis``, so its scale is arbitrary. It is compared
with VICTRE's own FBP by Pearson correlation inside the breast. OS-SART follows
the dataset's ground-truth recipe (5 interleaved subsets, nonnegative, invalid
rays masked, 8 extra 1 mm slices above and below), but it is fitted to the even
views only, so the 12 odd views measure prediction of unseen data.

The controls repeat both reconstructions with the source above the detector
centre and with the view order reversed. A correct geometry should agree better
with VICTRE's FBP and predict the held-out views better than either control.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torchtomo.victre import N_VIEWS, SOURCE_TO_DETECTOR_MM, victre_angles, victre_projector

GEOMETRIES = ("published", "source_over_detector_centre", "reversed_view_order")
TRAINING_VIEWS = list(range(0, N_VIEWS, 2))
HELD_OUT_VIEWS = list(range(1, N_VIEWS, 2))


def projector(geometry: str, density: str, bin_factor: int, views: list[int], margin: int, **options):
    overrides = {}
    if geometry == "source_over_detector_centre":
        overrides["source_y"] = 0.0
    elif geometry == "reversed_view_order":
        overrides["angles"] = victre_angles(device=options.get("device")).flip(0)[views]
    elif geometry != "published":
        raise ValueError(f"unknown geometry {geometry}")
    return victre_projector(density, bin_factor=bin_factor, views=views, depth_margin=margin, **options, **overrides)


def hann_ramp_filter(projections: torch.Tensor, operator) -> torch.Tensor:
    """Cosine weighting from the 0 degree source, then a Hann ramp along detector rows."""
    detector = operator.detector_positions[0]
    tube_travel = detector[..., 0]
    from_source = detector[..., 1] - operator.source_positions[0, 1]
    weights = SOURCE_TO_DETECTOR_MM / (SOURCE_TO_DETECTOR_MM**2 + tube_travel**2 + from_source**2).sqrt()
    weighted = projections * weights
    rows = weighted.shape[-2]
    size = 2 ** math.ceil(math.log2(rows * 1.6))
    low = (size - rows) // 2
    padded = F.pad(weighted.flatten(0, 2).transpose(-1, -2), (low, size - rows - low), mode="replicate")
    frequency = torch.fft.rfftfreq(size, device=projections.device)
    ramp = 2 * frequency * (0.5 + 0.5 * torch.cos(math.pi * frequency / frequency.max()))
    filtered = torch.fft.irfft(torch.fft.rfft(padded, dim=-1) * ramp, n=size, dim=-1)[..., low : low + rows]
    return filtered.transpose(-1, -2).reshape(weighted.shape)


def os_sart(subsets, iterations: int, shape: tuple[int, ...], device) -> torch.Tensor:
    """One SIRT step per subset, as ASTRA's SIRT3D_CUDA with a sinogram mask and MinConstraint 0."""
    volume = torch.zeros(1, 1, *shape, device=device)
    prepared = []
    for operator, measured, valid in subsets:
        row_sums = operator(torch.ones_like(volume))
        column_sums = operator.adjoint(valid.float())
        row_weight = torch.where(row_sums > 1e-6, 1 / row_sums, 0) * valid
        column_weight = torch.where(column_sums > 1e-6, 1 / column_sums, 0)
        prepared.append((operator, measured, row_weight, column_weight))
    for _ in range(iterations):
        for operator, measured, row_weight, column_weight in prepared:
            residual = (measured - operator(volume)) * row_weight
            volume = (volume + operator.adjoint(residual) * column_weight).clamp_min(0)
    return volume


def correlation(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    x = left[mask].double()
    y = right[mask].double()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()))


def relative_residual(predicted: torch.Tensor, measured: torch.Tensor, valid: torch.Tensor) -> float:
    return float(((predicted - measured) * valid).norm() / (measured * valid).norm())


def breast_mask(operator, projections, valid, reference) -> torch.Tensor:
    """Space carving from air rays, and VICTRE's FBP above 8% of its breast p99.5, as in the dataset."""
    outside = operator.adjoint(((projections <= 0) & valid).float())
    coverage = operator.adjoint(valid.float())
    hull = (outside <= 1e-3 * coverage.max()) & (coverage > 0.5 * coverage.max())
    smooth = F.avg_pool3d(reference, (1, 5, 5), stride=1, padding=(0, 2, 2), count_include_pad=False)
    level = torch.quantile(smooth[hull].float()[:: max(1, int(hull.sum()) // 2_000_000)], 0.995)
    return hull & (smooth > 0.08 * level)


def save_figure(volumes: dict[str, torch.Tensor], mask: torch.Tensor, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slices, columns = mask.shape[2], mask.shape[3]
    figure, axes = plt.subplots(2, len(volumes), figsize=(3.2 * len(volumes), 7.2), constrained_layout=True)
    for column, (name, volume) in enumerate(volumes.items()):
        values = volume[mask].cpu().float()
        low, high = torch.quantile(values[:: max(1, values.numel() // 1_000_000)], torch.tensor([0.01, 0.99]))
        image = volume[0, 0].cpu().numpy()
        axial = image[slices // 2].T
        depth = image[:, columns // 3].T
        axes[0, column].imshow(axial, cmap="gray", vmin=low.item(), vmax=high.item())
        axes[1, column].imshow(depth, cmap="gray", vmin=low.item(), vmax=high.item(), aspect="auto")
        axes[0, column].set_title(name, fontsize=9)
        for axis in axes[:, column]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_ylabel("middle slice: tube travel by chest-wall distance")
    axes[1, 0].set_ylabel("one third from the chest wall: tube travel by depth")
    figure.suptitle(title)
    figure.savefig(path, dpi=150)
    plt.close(figure)


@torch.no_grad()
def reconstruct(args) -> None:
    device = torch.device(args.device)
    data = np.load(args.input, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    density, bin_factor = metadata["density"], metadata["bin_factor"]
    pixel = 0.085 * bin_factor
    projections = torch.as_tensor(data["projections"], device=device)[None, None]
    valid = torch.as_tensor(data["valid"], device=device)[None, None]
    reference = torch.as_tensor(data["reference"], device=device).transpose(-1, -2)[None, None].contiguous()
    margin = args.depth_margin
    depth = reference.shape[2] + 2 * margin
    n_samples = args.n_samples or math.ceil(1.3 * depth / pixel)
    options = dict(n_samples=n_samples, ray_chunk=args.ray_chunk, device=device)
    every_view = list(range(N_VIEWS))
    published = projector("published", density, bin_factor, every_view, 0, **options)
    mask = breast_mask(published, projections, valid, reference)
    results = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "patient": metadata["patient"],
        "density": density,
        "bin_factor": bin_factor,
        "detector_shape": list(projections.shape[-2:]),
        "volume_shape_zyx": list(reference.shape[2:]),
        "voxel_mm_zyx": [1.0, pixel, pixel],
        "n_samples": n_samples,
        "mask_voxels": int(mask.sum()),
        "training_views": TRAINING_VIEWS,
        "held_out_views": HELD_OUT_VIEWS,
        "sart": {"iterations": args.sart_iterations, "subsets": args.subsets, "depth_margin_slices": margin},
        "geometries": {},
    }
    volumes = {"VICTRE FBP": reference}
    for geometry in args.geometries:
        entry = {}
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        full = projector(geometry, density, bin_factor, every_view, 0, **options)
        fbp = full.adjoint(hann_ramp_filter(projections * valid, full))
        if device.type == "cuda":
            torch.cuda.synchronize()
        entry["fbp_seconds"] = time.perf_counter() - started
        entry["fbp_reference_correlation"] = correlation(fbp, reference, mask)
        volumes[f"FBP, {geometry}"] = fbp
        if args.sart_iterations:
            started = time.perf_counter()
            subsets = []
            for start in range(args.subsets):
                views = TRAINING_VIEWS[start :: args.subsets]
                operator = projector(geometry, density, bin_factor, views, margin, **options)
                subsets.append((operator, projections[:, :, views], valid[:, :, views]))
            extended = os_sart(subsets, args.sart_iterations, (depth, *reference.shape[3:]), device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            entry["sart_seconds"] = time.perf_counter() - started
            training = projector(geometry, density, bin_factor, TRAINING_VIEWS, margin, **options)
            held_out = projector(geometry, density, bin_factor, HELD_OUT_VIEWS, margin, **options)
            entry["sart_training_relative_residual"] = relative_residual(
                training(extended), projections[:, :, TRAINING_VIEWS], valid[:, :, TRAINING_VIEWS]
            )
            entry["sart_held_out_relative_residual"] = relative_residual(
                held_out(extended), projections[:, :, HELD_OUT_VIEWS], valid[:, :, HELD_OUT_VIEWS]
            )
            sart = extended[:, :, margin : margin + reference.shape[2]]
            entry["sart_reference_correlation"] = correlation(sart, reference, mask)
            entry["sart_breast_median_per_mm"] = float(sart[mask].median())
            volumes[f"OS-SART, {geometry}"] = sart
        if device.type == "cuda":
            entry["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        results["geometries"][geometry] = entry
        print(json.dumps({geometry: entry}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    shown = {name: volume for name, volume in volumes.items() if "reversed" not in name}
    title = f"VICTRE {metadata['patient']} ({density}), {pixel:.2f} mm detector bins"
    save_figure(shown, mask, args.output / "reconstruction.png", title)
    if args.save_volumes:
        np.savez_compressed(
            args.output / "volumes.npz",
            mask=mask[0, 0].cpu().numpy(),
            **{name: volume[0, 0].cpu().numpy() for name, volume in volumes.items()},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometries", nargs="+", choices=GEOMETRIES, default=list(GEOMETRIES))
    parser.add_argument("--sart-iterations", type=int, default=10, help="0 skips OS-SART")
    parser.add_argument("--subsets", type=int, default=5)
    parser.add_argument("--depth-margin", type=int, default=8)
    parser.add_argument("--n-samples", type=int, help="default: 1.3 x depth / detector bin")
    parser.add_argument("--ray-chunk", type=int, default=16384)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-volumes", action="store_true")
    reconstruct(parser.parse_args())
