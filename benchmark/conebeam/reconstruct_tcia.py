"""Native 3D reconstruction of prepared TCIA helical rays, without fan beam rebinning."""

import argparse
import itertools
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from torchtomo import ConeBeam


def reference_volume(
    data: Mapping[str, np.ndarray], metadata: dict, shape: tuple, spacing: tuple, device: str
) -> torch.Tensor:
    axes = [
        (torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2) * step
        for size, step in zip(shape, spacing)
    ]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    reference_spacing = metadata["reference_spacing_zyx"]
    center_xy = metadata["reference_center_xy"]
    image_shape = data["reference_hu"].shape
    grid_x = (x - center_xy[0]) / (reference_spacing[2] * (image_shape[2] - 1) / 2)
    grid_y = (y - center_xy[1]) / (reference_spacing[1] * (image_shape[1] - 1) / 2)
    z_range = metadata["reference_z_range"]
    grid_z = 2 * (z + metadata["target_z"] - z_range[0]) / (z_range[1] - z_range[0]) - 1
    grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)[None]
    water = metadata["first_view"]["water_attenuation"]
    attenuation = ((torch.as_tensor(data["reference_hu"], device=device) / 1000 + 1) * water).clamp_min(0)
    return functional.grid_sample(attenuation[None, None], grid, align_corners=True)


def transform_positions(positions: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return positions @ matrix.T


def projection_metrics(predicted: torch.Tensor, measured: torch.Tensor) -> dict:
    x, y = predicted.flatten().double(), measured.flatten().double()
    relative = float((x - y).norm() / y.norm())
    centered_x, centered_y = x - x.mean(), y - y.mean()
    correlation = float((centered_x * centered_y).sum() / (centered_x.norm() * centered_y.norm()).clamp_min(1e-20))
    gain = float((x * y).sum() / x.square().sum().clamp_min(1e-20))
    return {"relative_error": relative, "correlation": correlation, "measured_over_predicted_gain": gain}


def image_metrics(reconstruction: torch.Tensor, reference: torch.Tensor, water: float) -> dict:
    ours = 1000 * (reconstruction[0, 0] / water - 1)
    truth = 1000 * (reference[0, 0] / water - 1)
    central = slice(ours.shape[0] // 2 - 2, ours.shape[0] // 2 + 2)
    ours, truth = ours[central], truth[central]
    mask = truth > -500
    x, y = ours[mask].double(), truth[mask].double()
    correlation = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()).norm() * (y - y.mean()).norm()))
    window_error = ((x.clamp(-160, 240) - y.clamp(-160, 240)) / 400).square().mean()
    return {
        "central_four_slices_body_correlation": correlation,
        "central_four_slices_body_hu_bias": float((x - y).mean()),
        "central_four_slices_body_hu_rmse": float((x - y).square().mean().sqrt()),
        "central_four_slices_body_window_psnr_db": float(-10 * window_error.log10()),
        "window_hu": [-160, 240],
        "body_threshold_hu": -500,
        "comparison": "vendor CT resampled onto the reconstruction grid, no image registration or intensity fitting",
    }


def save_figure(reconstruction: torch.Tensor, reference: torch.Tensor, water: float, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ours = (1000 * (reconstruction[0, 0] / water - 1)).cpu().numpy()
    truth = (1000 * (reference[0, 0] / water - 1)).cpu().numpy()
    planes = [(0, ours.shape[0] // 2), (1, ours.shape[1] // 2), (2, ours.shape[2] // 2)]
    figure, axes = plt.subplots(3, 3, figsize=(10, 8), constrained_layout=True)
    for column, (dimension, index) in enumerate(planes):
        original = np.take(truth, index, axis=dimension)
        reconstructed = np.take(ours, index, axis=dimension)
        for row, image in enumerate((original, reconstructed, reconstructed - original)):
            image_artist = axes[row, column].imshow(
                image,
                cmap="gray" if row < 2 else "coolwarm",
                vmin=-160 if row < 2 else -200,
                vmax=240 if row < 2 else 200,
                aspect=1.0 if dimension == 0 else 4.0 / (500 / 128),
                origin="upper" if dimension == 0 else "lower",
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[0, column].set_title(("Axial", "Coronal", "Sagittal")[column])
    for row, name in enumerate(("Vendor CT", "3D CGLS", "Difference, HU")):
        axes[row, 0].set_ylabel(name)
    figure.colorbar(image_artist, ax=axes[2, :], orientation="horizontal", shrink=0.7, label="Difference, HU")
    figure.suptitle("TCIA L213, native helical rays, cylindrical detector\nCT window: -160 to 240 HU")
    figure.savefig(output / "reconstruction.png", dpi=160)
    figure.savefig(output / "reconstruction.svg")
    plt.close(figure)


@torch.no_grad()
def reconstruct(input_path: Path, output: Path, iterations: int, device: str, backend: str = "auto") -> None:
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(input_path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    shape, spacing = (32, 128, 128), (4.0, 500 / 128, 500 / 128)
    center = (0.0, 0.0, metadata["target_z"])
    sources = torch.as_tensor(data["sources"], device=device)
    detectors = torch.as_tensor(data["detectors"], device=device)
    measured = torch.as_tensor(data["projections"], device=device)[None, None]
    reference = reference_volume(data, metadata, shape, spacing, device)
    training = torch.arange(0, sources.shape[0], 2, device=device)
    held_out = torch.arange(1, sources.shape[0], 2, device=device)
    calibration = training[torch.linspace(0, len(training) - 1, 8, device=device).round().long()]

    def projector(selected: torch.Tensor, matrix: torch.Tensor) -> ConeBeam:
        return ConeBeam(
            shape,
            transform_positions(sources[selected], matrix),
            transform_positions(detectors[selected], matrix),
            spacing,
            center,
            n_samples=192,
            ray_chunk=2048,
            backend=backend,
        )

    conventions = []
    matrices = []
    for reflected, quarter_turn in itertools.product((False, True), range(4)):
        angle = quarter_turn * math.pi / 2
        matrix = torch.tensor(
            [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]], device=device
        )
        if reflected:
            matrix[:, 0] *= -1
        scores = projection_metrics(projector(calibration, matrix)(reference), measured[:, :, calibration])
        conventions.append({"reflected_x": reflected, "quarter_turn": quarter_turn, **scores})
        matrices.append(matrix)
        print("frame", json.dumps(conventions[-1]), flush=True)
    best = min(range(len(conventions)), key=lambda index: conventions[index]["relative_error"])
    selected_matrix = matrices[best]
    operator = projector(training, selected_matrix)
    validation_operator = projector(held_out, selected_matrix)
    observations = measured[:, :, training]
    validation_observations = measured[:, :, held_out]
    reference_metrics = projection_metrics(operator(reference), observations)
    print("vendor forward check", json.dumps(reference_metrics), flush=True)
    if reference_metrics["correlation"] < 0.85 or reference_metrics["relative_error"] > 0.4:
        raise RuntimeError("vendor forward check failed; inspect geometry before reconstruction")
    volume = torch.zeros_like(reference)
    residual = observations.clone()
    gradient = operator.adjoint(residual)
    direction = gradient.clone()
    gamma = gradient.square().sum()
    history = []
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for iteration in range(1, iterations + 1):
        projected = operator(direction)
        alpha = gamma / projected.square().sum().clamp_min(1e-20)
        volume += alpha * direction
        residual -= alpha * projected
        next_gradient = operator.adjoint(residual)
        next_gamma = next_gradient.square().sum()
        direction = next_gradient + next_gamma / gamma.clamp_min(1e-20) * direction
        gamma = next_gamma
        entry = {"iteration": iteration, "training_relative_residual": float(residual.norm() / observations.norm())}
        if iteration == 1 or iteration % 5 == 0 or iteration == iterations:
            entry["held_out"] = projection_metrics(validation_operator(volume), validation_observations)
            entry["vendor_comparison"] = image_metrics(volume, reference, metadata["first_view"]["water_attenuation"])
        history.append(entry)
        print(json.dumps(entry), flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    results = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "input_metadata": metadata,
        "volume_shape_zyx": shape,
        "voxel_spacing_zyx_mm": spacing,
        "volume_center_xyz_mm": center,
        "frame_candidates": conventions,
        "selected_frame": conventions[best],
        "frame_selection_uses_vendor_ct": True,
        "training_views": len(training),
        "held_out_views": len(held_out),
        "vendor_forward_check": reference_metrics,
        "cgls_seconds_including_validation": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
        "history": history,
        "limitations": [
            "one full-dose patient, coarse grid and sparse detector/view sampling",
            "vendor images were used to select a planar coordinate frame",
            "unregularized CGLS, no statistical weighting or scanner-specific corrections",
        ],
    }
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(output / "volumes.npz", reconstruction=volume.cpu().numpy(), reference=reference.cpu().numpy())
    save_figure(volume, reference, metadata["first_view"]["water_attenuation"], output)
    print("DONE", json.dumps({"output": str(output), **history[-1]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--backend", choices=("auto", "torch", "cuda"), default="auto")
    args = parser.parse_args()
    reconstruct(args.input, args.output, args.iterations, args.device, args.backend)
