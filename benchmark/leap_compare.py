"""Score LEAP's parallel-beam projector and its untrained methods on the example's data.

This reads the dataset a run of examples/ellipses/train.py wrote, reconstructs the
same slices from the same noisy sinograms with LEAP, and reports the same PSNRs, so
the rows drop straight into the table that run produced.

LEAP is an optional benchmark dependency, like torch-radon. Nothing in the library
or in examples/ellipses imports it.

    PYTHONPATH=src:examples/ellipses python benchmark/leap_compare.py \
        --results examples/ellipses/results-ctw
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from data import apply_window, psnr_per_image
from leapctype import tomographicModels
from train import summarize

from torchtomo import ParallelBeam

LOGGER = logging.getLogger("leap")


def leap_parallel_beam(size, angles, slices, gpu=0):
    """A LEAP geometry matching torchtomo's: image on [-1, 1], detector likewise.

    LEAP turns the gantry the other way round, so the arc is negated; with that one
    change the two forward projectors agree to about 0.1% in relative L2.
    """
    pixel = 2.0 / size
    lct = tomographicModels()
    lct.set_gpu(gpu)
    lct.print_warnings = False
    phis = np.ascontiguousarray(-np.degrees(angles.detach().cpu().numpy()), dtype=np.float32)
    lct.set_parallelbeam(len(phis), slices, size, pixel, pixel, (slices - 1) / 2.0, (size - 1) / 2.0, phis)
    lct.set_volume(size, size, slices, pixel, pixel)
    return lct


def to_leap_sinogram(sinogram):
    """[B, 1, angles, detectors] -> LEAP's [angles, rows, detectors], one row per slice."""
    return sinogram[:, 0].permute(1, 0, 2).contiguous()


def to_leap_volume(image):
    """[B, 1, H, W] -> LEAP's [slices, y, x]."""
    return image[:, 0].contiguous()


def from_leap_volume(volume):
    return volume.unsqueeze(1)


def leap_fbp(lct, sinogram, mask, callback=None):
    volume = torch.zeros(sinogram.shape[1], sinogram.shape[2], sinogram.shape[2], device=sinogram.device)
    lct.FBP(sinogram.clone(), volume)
    # No clamp: torchtomo's fbp() does not clamp either, and the table compares the two.
    image = from_leap_volume(volume) * mask
    if callback is not None:
        callback(1, image)
    return image


def leap_sart(lct, sinogram, mask, iterations, subsets=1, callback=None):
    """LEAP's own SART, one iteration at a time so the whole curve is scored.

    SART updates the volume in place from whatever it is given, so stepping it
    repeatedly is the same run as asking for every iteration at once.
    """
    size = sinogram.shape[2]
    volume = torch.zeros(sinogram.shape[1], size, size, device=sinogram.device)
    for step in range(1, iterations + 1):
        lct.SART(sinogram, volume, 1, subsets)
        if callback is not None:
            callback(step, from_leap_volume(volume) * mask)
    return from_leap_volume(volume) * mask


def score_curve(reconstruct, sinograms, truth, windows, region, batch_size, device):
    """Mean PSNR after every iteration, in the convention the example selects on."""
    totals = {}
    window_batches = windows.split(batch_size) if windows is not None else [None] * len(sinograms.split(batch_size))
    for sinogram, target, window in zip(sinograms.split(batch_size), truth.split(batch_size), window_batches):
        target = target.to(device)
        window = None if window is None else window.to(device)

        def record(step, image, target=target, window=window):
            if window is None:
                scores = psnr_per_image(image, target, region=region)
            else:
                scores = psnr_per_image(apply_window(image, window), apply_window(target, window), region=region)
            totals[step] = totals.get(step, 0.0) + scores.sum().item()

        reconstruct(sinogram.to(device), record)
    return [{"iteration": step, "psnr_db": total / len(sinograms)} for step, total in sorted(totals.items())]


def reconstruct_split(reconstruct, sinograms, batch_size, device):
    return torch.cat([reconstruct(batch.to(device), None).cpu() for batch in sinograms.split(batch_size)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="A directory written by train.py")
    parser.add_argument("--output", type=Path, default=None, help="Where to write leap-metrics.json; default --results")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sirt-iterations", type=int, default=300)
    parser.add_argument("--sart-updates", type=int, default=300)
    parser.add_argument("--sart-subsets", type=int, default=0, help="0 uses one view per subset, as the example does")
    parser.add_argument("--limit", type=int, default=0, help="Score only this many slices per split; 0 uses all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = args.output or args.results
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.results / "config.json").read_text())
    data = torch.load(args.results / "dataset.pt", map_location="cpu")
    size, angles = config["image_size"], config["angles"]
    device = torch.device(args.device)
    projector = ParallelBeam(img_size=size, n_angles=angles).to(device)
    region = projector.circle_mask.bool()
    mask = projector.circle_mask[None, None]
    val_ids, test_ids = data["splits"]["val"], data["splits"]["test"]
    if args.limit:
        val_ids, test_ids = val_ids[: args.limit], test_ids[: args.limit]
    windows = data.get("windows")
    subsets = args.sart_subsets or angles
    LOGGER.info(
        "leap results=%s size=%d angles=%d val=%d test=%d batch=%d sart_subsets=%d",
        args.results,
        size,
        angles,
        len(val_ids),
        len(test_ids),
        args.batch_size,
        subsets,
    )

    # One geometry per batch shape: LEAP carries the slice count in the geometry.
    geometries = {}

    def projector_for(count):
        if count not in geometries:
            geometries[count] = leap_parallel_beam(size, projector.angles, count, gpu=0)
        return geometries[count]

    def fbp_run(sinogram, callback):
        lct = projector_for(sinogram.shape[0])
        return leap_fbp(lct, to_leap_sinogram(sinogram), mask, callback)

    def sart_run(iterations, subset_count):
        def run(sinogram, callback):
            lct = projector_for(sinogram.shape[0])
            return leap_sart(lct, to_leap_sinogram(sinogram), mask, iterations, subset_count, callback)

        return run

    methods = {
        "leap-fbp": (fbp_run, 1),
        "leap-sirt": (sart_run(args.sirt_iterations, 1), args.sirt_iterations),
        "leap-sart": (sart_run(args.sart_updates, subsets), args.sart_updates),
    }

    selection, reconstructions, timings = {}, {}, {}
    for name, (run, budget) in methods.items():
        started = time.perf_counter()
        curve = score_curve(
            run,
            data["noisy"][val_ids],
            data["truth"][val_ids],
            None if windows is None else windows[val_ids],
            region,
            args.batch_size,
            device,
        )
        top = max(curve, key=lambda row: row["psnr_db"])
        LOGGER.info(
            "leap method=%s budget=%d best_val_iteration=%d val_psnr_db=%.4f search_seconds=%.1f",
            name,
            budget,
            top["iteration"],
            top["psnr_db"],
            time.perf_counter() - started,
        )
        selection[name] = {"curve": curve, "selected_iteration": top["iteration"], "val_psnr_db": top["psnr_db"]}
        chosen = fbp_run if name == "leap-fbp" else sart_run(top["iteration"], 1 if name == "leap-sirt" else subsets)
        started = time.perf_counter()
        reconstructions[name] = reconstruct_split(chosen, data["noisy"][test_ids], args.batch_size, device)
        timings[name] = time.perf_counter() - started

    test_windows = windows[test_ids] if windows is not None else None
    metrics = {}
    for name, prediction in reconstructions.items():
        metrics[name] = summarize(prediction, data["truth"][test_ids], region=region, windows=test_windows)
        metrics[name]["test_reconstruction_seconds"] = timings[name]
        metrics[name]["selected_iteration"] = selection[name]["selected_iteration"]
        LOGGER.info(
            "TEST method=%s mean_psnr_db=%.4f roi_psnr_db=%.4f window_psnr_db=%s",
            name,
            metrics[name]["mean_psnr_db"],
            metrics[name]["mean_roi_psnr_db"],
            f"{metrics[name]['mean_window_psnr_db']:.4f}" if test_windows is not None else "n/a",
        )

    (output / "leap-selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (output / "leap-metrics.json").write_text(
        json.dumps({"split": "test", "ids": test_ids.tolist(), "methods": metrics}, indent=2) + "\n"
    )
    torch.save({"ids": test_ids, **reconstructions}, output / "leap-reconstructions.pt")

    reference = json.loads((args.results / "metrics.json").read_text())["methods"]
    key = "mean_window_psnr_db" if test_windows is not None else "mean_roi_psnr_db"
    LOGGER.info("comparison metric=%s", key)
    for leap_name, torchtomo_name in (("leap-fbp", "fbp"), ("leap-sirt", "sirt"), ("leap-sart", "sart")):
        if torchtomo_name in reference:
            LOGGER.info(
                "%-10s leap=%.4f torchtomo=%.4f difference=%+.4f",
                torchtomo_name,
                metrics[leap_name][key],
                reference[torchtomo_name][key],
                metrics[leap_name][key] - reference[torchtomo_name][key],
            )


if __name__ == "__main__":
    main()
