"""Compare FBP, SIRT, SART, FBP+U-Net, RED, and LPD on ellipse phantoms or real CT slices.

The U-Net and LPD are trained here. SIRT, SART, and RED are not: they are stopped
early at a point chosen on the validation split, and RED reuses the trained U-Net.

From the repository root:
    PYTHONPATH=src .venv/bin/python examples/ellipses/train.py
"""

import argparse
import json
import logging
import os
import platform
import time
from pathlib import Path

import torch
from classical import run_classical
from data import (
    apply_in_batches,
    apply_window,
    calibrate_photons,
    load_ct_splits,
    make_phantoms,
    make_splits,
    poisson_sinogram,
    psnr_per_image,
)
from models import FBPUNet, IRadonMap, LearnedPrimalDual, estimate_operator_norm
from objectives import Noise2Inverse, Proj2Proj, Supervised
from torch.nn import functional as F

from torchtomo import FanBeam, ParallelBeam

LOGGER = logging.getLogger("ellipses")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def fbp_split_seconds(projector, sinograms, device, batch_size):
    """Wall-clock of FBP on one split, on the device the rest of the run uses."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    images = torch.cat([projector.fbp(batch.to(device)).cpu() for batch in sinograms.split(batch_size)])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return images, time.perf_counter() - start


def record_method_seconds(metrics, training, selection, fbp_seconds):
    """Copy wall-clock seconds onto every scored method in metrics.json.

    Learned methods use training_seconds from train_model; classical methods use
    the search-plus-reconstruct time recorded in run_classical; FBP uses the
    timed reconstruction of the scored split.
    """
    for name, row in metrics.items():
        if name == "fbp":
            seconds = fbp_seconds
        elif name in training:
            seconds = training[name]["training_seconds"]
        elif name in selection:
            seconds = selection[name]["seconds"]
        else:
            raise KeyError(f"no seconds recorded for method {name!r}")
        seconds = float(seconds)
        if not (seconds > 0 and seconds != float("inf")):
            raise ValueError(f"{name} seconds must be a positive finite value, got {seconds}")
        row["seconds"] = seconds


def save_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def model_weights(model):
    # The large fixed geometry is recreated from config, not copied each epoch.
    return {key: value for key, value in model.state_dict().items() if not key.startswith("projector.")}


def load_weights(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("projector.") for key in missing):
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def setup_logging(output):
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output / "training.log", mode="a")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def build_truth(args, projector, output):
    """Return ground truth, split indices, and a provenance record.

    Real slices are scaled by the training split's maximum so attenuation stays
    comparable across patients, and masked to the circle the geometry can see.
    """
    if args.data_dir is None:
        truth, ellipses = make_phantoms(size=args.image_size, seed=args.seed)
        splits = make_splits(args.seed + 1)
        assert [len(splits[key]) for key in ("train", "val", "test")] == [60, 20, 20]
        assert len(set(torch.cat(list(splits.values())).tolist())) == 100
        write_json(
            output / "phantoms.json",
            {"parameter_order": ["intensity", "a", "b", "cx", "cy", "radians"], "ellipses": ellipses},
        )
        source, windows = {"source": "generated ellipses", "seed": args.seed}, None
    else:
        truth, splits, names, windows = load_ct_splits(args.data_dir, args.image_size)
        scale = truth[splits["train"]].max().item()
        truth = truth / scale
        if windows is not None:
            windows = windows / scale  # the window lives in the same units as the images
        write_json(
            output / "slices.json",
            {
                "source": str(Path(args.data_dir).resolve()),
                "attenuation_scale": scale,
                "names": {key: [names[i] for i in value.tolist()] for key, value in splits.items()},
            },
        )
        source = {"source": str(Path(args.data_dir).resolve()), "attenuation_scale": scale}
        LOGGER.info("loaded real CT slices scale=%.6f from %s", scale, args.data_dir)

    # The projector cannot see outside the inscribed circle, so no method should be
    # scored on content there. Ellipse phantoms already stop short of it.
    truth = truth * projector.circle_mask.view(1, 1, args.image_size, args.image_size)
    return truth, splits, source, windows


def build_projector(args):
    """The geometry, from whichever backend the run asked for.

    LEAP is an optional benchmark dependency, so it is imported only when chosen;
    an ordinary run never touches it. Fan beam uses the size-aware FanBeam defaults
    (src = det = 2 * img_size, 1.5 bins per pixel at the isocentre) unless overridden.
    """
    if getattr(args, "geometry", "parallel") == "fan":
        kwargs = dict(img_size=args.image_size, n_angles=args.angles)
        if args.src_dist is not None:
            kwargs["src_dist"] = args.src_dist
        if args.det_dist is not None:
            kwargs["det_dist"] = args.det_dist
        if args.n_det is not None:
            kwargs["n_det"] = args.n_det
        if args.projector == "leap":
            from leap_projector import LeapFanBeam

            return LeapFanBeam(**kwargs)
        return FanBeam(**kwargs, backend=getattr(args, "backend", "torch"))
    if args.projector == "leap":
        from leap_projector import LeapParallelBeam

        return LeapParallelBeam(img_size=args.image_size, n_angles=args.angles)
    return ParallelBeam(img_size=args.image_size, n_angles=args.angles, backend=getattr(args, "backend", "torch"))


def prepare_data(args, output):
    projector = build_projector(args)
    truth, splits, source, windows = build_truth(args, projector, output)
    sizes = {key: len(value) for key, value in splits.items()}
    clean = apply_in_batches(projector.forward, truth, args.batch_size)
    clean_fbp = apply_in_batches(projector.fbp, clean, args.batch_size)
    # The dose-free limit of the analytic method, in every convention the table uses,
    # so it is never compared against a figure measured over a different region.
    ceiling = summarize(
        clean_fbp[splits["train"]],
        truth[splits["train"]],
        region=projector.circle_mask.bool(),
        windows=None if windows is None else windows[splits["train"]],
    )
    LOGGER.info(
        "clean train FBP psnr_db=%.4f roi_psnr_db=%.4f window_psnr_db=%s",
        ceiling["mean_psnr_db"],
        ceiling["mean_roi_psnr_db"],
        f"{ceiling['mean_window_psnr_db']:.4f}" if windows is not None else "n/a",
    )
    if args.photons is not None:
        # A dose taken from the dataset's own photon grid, rather than fitted to a target.
        photons, calibration = args.photons, []
        LOGGER.info("photons fixed at %.3f; calibration skipped", photons)
    else:
        photons, calibration = calibrate_photons(
            projector,
            clean[splits["train"]],
            truth[splits["train"]],
            target=args.target_psnr,
            seed=args.seed + 2,
            windows=None if windows is None else windows[splits["train"]],
        )
    # A fresh fixed realization is shared by all three methods, after calibration.
    noisy, counts = poisson_sinogram(clean, photons, args.seed + 3)
    fbp = apply_in_batches(projector.fbp, noisy, args.batch_size)
    data = {"truth": truth, "clean": clean, "noisy": noisy, "counts": counts, "fbp": fbp, "splits": splits}
    if windows is not None:
        data["windows"] = windows
    torch.save(data, output / "dataset.pt")
    write_json(output / "splits.json", {key: value.tolist() for key, value in splits.items()})
    write_json(
        output / "noise-calibration.json", {"split": "train", "trials": calibration, "selected_photons": photons}
    )
    LOGGER.info(
        "dataset %s size=%d split=%d/%d/%d photons=%.3f zero_count_fraction=%.8f",
        source["source"],
        sum(sizes.values()),
        sizes["train"],
        sizes["val"],
        sizes["test"],
        photons,
        (counts == 0).float().mean(),
    )
    # Do not evaluate the test split until after model/checkpoint selection.
    for name in ("train", "val"):
        ids = splits[name]
        windowed = "n/a"
        if windows is not None:
            scored = psnr_per_image(apply_window(fbp[ids], windows[ids]), apply_window(truth[ids], windows[ids]))
            windowed = f"{scored.mean():.4f}"
        LOGGER.info(
            "baseline split=%s fbp_psnr_db=%.4f fbp_window_psnr_db=%s",
            name,
            psnr_per_image(fbp[ids], truth[ids]).mean(),
            windowed,
        )
    return projector, data, photons, source


def summarize(prediction, truth, region=None, windows=None):
    """Score a reconstruction, optionally also over the visible circle only.

    The full-image figure includes the corners every method zeroes out, which
    flatters all of them by about 1 dB, so the region figure is the honest one.
    """
    scores = psnr_per_image(prediction, truth)
    result = {
        "mse": F.mse_loss(prediction, truth).item(),
        "mean_psnr_db": scores.mean().item(),
        "std_psnr_db": scores.std(unbiased=False).item(),
        "per_image_psnr_db": scores.tolist(),
    }
    if region is not None:
        roi = psnr_per_image(prediction, truth, region=region)
        result["mean_roi_psnr_db"] = roi.mean().item()
        result["std_roi_psnr_db"] = roi.std(unbiased=False).item()
        result["per_image_roi_psnr_db"] = roi.tolist()
    if windows is not None:
        scored = psnr_per_image(apply_window(prediction, windows), apply_window(truth, windows), region=region)
        result["mean_window_psnr_db"] = scored.mean().item()
        result["std_window_psnr_db"] = scored.std(unbiased=False).item()
        result["per_image_window_psnr_db"] = scored.tolist()
    return result


def train_model(name, model, objective, data, args, output, epochs, seed, learning_rate):
    device = torch.device(args.device)
    model.to(device)
    train_ids, val_ids = data["splits"]["train"], data["splits"]["val"]
    truth = data["truth"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)
    generator = torch.Generator().manual_seed(seed)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    LOGGER.info(
        "model=%s parameters=%d epochs=%d lr=%.3g device=%s supervised=%s selected_on=%s",
        name,
        parameter_count,
        epochs,
        learning_rate,
        device,
        objective.uses_truth,
        objective.criterion,
    )
    history, best_mse, best_epoch = [], float("inf"), 0
    first_epoch, previous_seconds = 1, 0.0
    last_path = output / f"{name}-last.pt"
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu")
        load_weights(model, checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        generator.set_state(checkpoint["shuffle_state"])
        history = checkpoint["history"]
        best_mse, best_epoch = checkpoint["best_mse"], checkpoint["best_epoch"]
        first_epoch = checkpoint["epoch"] + 1
        previous_seconds = checkpoint["training_seconds"]
        LOGGER.info("model=%s resuming_at_epoch=%d", name, first_epoch)
    peak_memory_bytes = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for epoch in range(first_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = train_ids[torch.randperm(len(train_ids), generator=generator)]
        squared_error = 0.0
        lr = optimizer.param_groups[0]["lr"]
        batches = list(order.split(args.batch_size))
        for step, ids in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            # A global counter, so the methods that cycle a mask or a held-out
            # subset keep advancing across epochs and resume where they stopped.
            global_step = (epoch - 1) * len(batches) + step
            batch_error = 0.0
            # Every objective's loss is a mean over the images in the call, so
            # weighting each piece by its share rebuilds the whole batch's gradient
            # exactly. Only the activations held at once shrink.
            for piece in ids.split(args.micro_batch or len(ids)):
                loss = objective.loss(model, piece, device, global_step)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite {name} loss at epoch {epoch}")
                (loss * (len(piece) / len(ids))).backward()
                batch_error += loss.item() * len(piece)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite {name} gradient at epoch {epoch}")
            if epoch == 1 and step == 0:
                LOGGER.info("model=%s initial_gradient_norm=%.8g", name, grad_norm)
                if isinstance(model, LearnedPrimalDual):
                    for branch in ("dual_updates", "primal_updates"):
                        first_block = getattr(model, branch)[0]
                        norm = (
                            sum(p.grad.square().sum().item() for p in first_block.parameters() if p.grad is not None)
                            ** 0.5
                        )
                        if norm == 0:
                            raise RuntimeError(f"LPD gradient does not reach the first {branch} block")
                        LOGGER.info("model=lpd first_%s_gradient_norm=%.8g", branch, norm)
            optimizer.step()
            squared_error += batch_error
        # Selection uses the objective's own criterion. For the self-supervised
        # methods that is a loss built from the measurements, so no clean image is
        # consulted; the PSNR beside it is recorded for the curves only.
        val_loss = objective.validation(model, val_ids, device, args.batch_size)
        reference = summarize(objective.reconstruct(model, val_ids, device, args.batch_size), truth[val_ids])
        record = {
            "epoch": epoch,
            "train_mse": squared_error / len(train_ids),
            "val_mse": val_loss,
            "val_psnr_db": reference["mean_psnr_db"],
            "learning_rate": lr,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        if val_loss < best_mse:
            best_mse, best_epoch = val_loss, epoch
            save_checkpoint(
                output / f"{name}-best.pt",
                {"state_dict": model_weights(model), "epoch": epoch, "val_mse": best_mse},
            )
        LOGGER.info(
            "model=%s epoch=%03d/%03d train_mse=%.7f val_mse=%.7f val_psnr_db=%.4f lr=%.3g seconds=%.2f best_epoch=%d",
            name,
            epoch,
            epochs,
            record["train_mse"],
            record["val_mse"],
            record["val_psnr_db"],
            lr,
            record["seconds"],
            best_epoch,
        )
        write_json(output / f"{name}-history.json", history)
        scheduler.step()
        save_checkpoint(
            last_path,
            {
                "state_dict": model_weights(model),
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "shuffle_state": generator.get_state(),
                "history": history,
                "best_mse": best_mse,
                "best_epoch": best_epoch,
                "training_seconds": previous_seconds + time.perf_counter() - start,
            },
        )
    checkpoint = torch.load(output / f"{name}-best.pt", map_location=device)
    load_weights(model, checkpoint["state_dict"])
    duration = previous_seconds + time.perf_counter() - start
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        LOGGER.info(
            "model=%s training_complete seconds=%.2f best_epoch=%d peak_memory_bytes=%d",
            name,
            duration,
            best_epoch,
            peak_memory_bytes,
        )
    else:
        LOGGER.info("model=%s training_complete seconds=%.2f best_epoch=%d", name, duration, best_epoch)
    summary = {
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "training_seconds": duration,
        "supervised": objective.uses_truth,
        "criterion": objective.criterion,
    }
    if peak_memory_bytes is not None:
        summary["peak_memory_bytes"] = peak_memory_bytes
    return summary


def save_plots(output, data, reconstructions, metrics, supervision, tile=256):
    # Matplotlib is already part of the repository's existing dev extra.
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_ids = data["splits"]["test"].tolist()
    truth = data["truth"][test_ids]
    # Real runs record a source file per slice; phantom runs fall back to the index.
    labels_by_index = {}
    slices = output / "slices.json"
    if slices.exists():
        names = json.loads(slices.read_text())["names"]["test"]
        labels_by_index = {index: name.split(".")[0] for index, name in zip(test_ids, names)}
    images = {"gt": truth, **reconstructions}
    # Render inside each slice's own display window when there is one, so soft tissue is
    # not squeezed into a fraction of the grey scale by bone at the top of the range.
    windows = data["windows"][test_ids] if "windows" in data else None
    if windows is not None:
        images = {name: apply_window(tensor, windows) for name, tensor in images.items()}
    # Figures are read at a few hundred pixels per panel, so a 512 grid of seven methods
    # would spend megabytes on detail no viewer sees. Metrics use the full resolution.
    factor = truth.shape[-1] // tile if tile else 0
    if factor > 1:
        images = {name: F.avg_pool2d(tensor, factor) for name, tensor in images.items()}
    labels = {
        "gt": "Ground truth",
        "fbp": "FBP",
        "sirt": "SIRT",
        "sart": "SART",
        "bm3d": "FBP + BM3D",
        "fbp-unet": "FBP + U-Net",
        "red": "RED (U-Net prior)",
        "iradonmap": "iRadonMAP",
        "noise2inverse": "Noise2Inverse",
        "proj2proj": "Proj2Proj",
        "lpd": "Learned Primal-Dual",
    }
    # Every PNG has the same ordering and [0, 1] display range. Metrics use raw values.
    # Large test splits are capped at the first 20 slices so the grids stay readable.
    columns = 5
    shown = min(len(truth), 20)
    filled = -(-shown // columns) * columns
    for name, tensor in images.items():
        tiles = tensor[:shown, 0].clamp(0, 1)
        if len(tiles) < filled:
            tiles = torch.cat([tiles, tiles.new_zeros(filled - len(tiles), *tiles.shape[1:])])
        rows = [torch.cat(list(tiles[start : start + columns]), dim=1) for start in range(0, filled, columns)]
        zoom = max(1, min(3, 192 // tensor.shape[-1]))
        grid = torch.cat(rows, dim=0).repeat_interleave(zoom, 0).repeat_interleave(zoom, 1)
        plt.imsave(output / f"{name}.png", grid.numpy(), cmap="gray", vmin=0, vmax=1)

    # Compare the leading held-out images, selected by split order, not by score.
    panels = min(4, len(truth))
    fig, axes = plt.subplots(
        panels, len(images), figsize=(2.5 * len(images), 2.5 * panels), constrained_layout=True, squeeze=False
    )
    for row in range(panels):
        for column, (name, tensor) in enumerate(images.items()):
            ax = axes[row, column]
            ax.imshow(tensor[row, 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if name == "gt":
                subtitle = labels_by_index.get(test_ids[row], f"Phantom {test_ids[row]:03d}")
            else:
                key = "per_image_window_psnr_db" if windows is not None else "per_image_psnr_db"
                subtitle = f"{metrics[name][key][row]:.2f} dB"
            ax.set_title(f"{labels[name]}\n{subtitle}", fontsize=10)
            ax.set_axis_off()
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)

    # The self-supervised losses are different quantities from an image MSE, so they
    # get their own panel rather than sharing an axis that would invite comparison.
    colors = {
        "fbp-unet": "#2764b0",
        "lpd": "#cf5b2e",
        "iradonmap": "#4b9b4b",
        "noise2inverse": "#8b5cb8",
        "proj2proj": "#c9a227",
    }
    trained = [name for name in colors if (output / f"{name}-history.json").exists()]
    paired = [name for name in trained if supervision.get(name, True)]
    blind = [name for name in trained if name not in paired]

    panels = 3 if blind else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4), constrained_layout=True)
    histories = {name: json.loads((output / f"{name}-history.json").read_text()) for name in trained}

    def draw_loss(ax, names, title):
        for name in names:
            history = histories[name]
            epochs = [row["epoch"] for row in history]
            ax.plot(epochs, [row["train_mse"] for row in history], "--", color=colors[name], linewidth=1)
            ax.plot(epochs, [row["val_mse"] for row in history], color=colors[name], label=labels[name])
            best = min(history, key=lambda row: row["val_mse"])
            ax.scatter(best["epoch"], best["val_mse"], color=colors[name], edgecolor="black", zorder=3)
        ax.set(ylabel="Selection criterion", yscale="log", title=title)

    draw_loss(axes[0], paired, "Image MSE against the clean image")
    if blind:
        draw_loss(axes[1], blind, "Self-supervised loss, no clean image")
    quality = axes[-1]
    for name in trained:
        history = histories[name]
        quality.plot(
            [row["epoch"] for row in history],
            [row["val_psnr_db"] for row in history],
            color=colors[name],
            label=labels[name],
        )
        best = min(history, key=lambda row: row["val_mse"])
        quality.scatter(best["epoch"], best["val_psnr_db"], color=colors[name], edgecolor="black", zorder=3)
    val_ids = data["splits"]["val"]
    baseline = psnr_per_image(data["fbp"][val_ids], data["truth"][val_ids]).mean().item()
    quality.axhline(baseline, color="#777777", linestyle=":", label=f"FBP validation ({baseline:.2f} dB)")
    quality.set(ylabel="Mean PSNR (dB)", title="Validation quality, marked at the selected epoch")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "curves.png", dpi=150)
    plt.close(fig)

    selection_path = output / "classical-selection.json"
    if selection_path.exists():
        save_selection_plot(output, json.loads(selection_path.read_text()), labels)


def save_selection_plot(output, selection, labels):
    """Show how each untrained method was stopped, using validation scores only."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(selection), figsize=(4.2 * len(selection), 3.6), constrained_layout=True, squeeze=False
    )
    for ax, (name, record) in zip(axes[0], selection.items()):
        if record["searched_steps"] == 1:
            # Nothing is iterated here, so the search runs over the setting itself.
            settings = [float(setting) for setting in record["curves"]]
            ax.plot(settings, [curve[0]["psnr_db"] for curve in record["curves"].values()], marker="o", linewidth=1.2)
            ax.scatter(record["selected_setting"], record["val_psnr_db"], color="black", zorder=3)
            ax.set(xlabel=record.get("setting_name", "setting").capitalize())
            ax.set(ylabel="Validation PSNR (dB)", title=labels[name])
            ax.grid(alpha=0.2)
            continue
        for setting, curve in record["curves"].items():
            steps = [row["iteration"] for row in curve]
            scores = [row["psnr_db"] for row in curve]
            ax.plot(steps, scores, label=setting if setting != "default" else None, linewidth=1.2)
        ax.scatter(
            record["selected_steps"],
            record["val_psnr_db"],
            color="black",
            zorder=3,
            label=f"Selected: {record['selected_steps']} {record['unit']}",
        )
        # Every method's useful range sits near the start, before noise is fitted.
        ax.set_xscale("log")
        ax.set(xlabel=record["unit"].capitalize(), ylabel="Validation PSNR (dB)", title=labels[name])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "classical-selection.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--angles", type=int, default=90)
    parser.add_argument("--unet-epochs", type=int, default=120)
    parser.add_argument("--lpd-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-psnr", type=float, default=23.0)
    parser.add_argument(
        "--photons",
        type=float,
        default=None,
        help="Fix the incident photon count instead of calibrating it to --target-psnr",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=0,
        help="Accumulate each training batch in pieces of this many images; 0 keeps the batch whole. "
        "The gradient is unchanged, so this only trades speed for peak memory on a smaller card",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--projector",
        choices=("torchtomo", "leap"),
        default="torchtomo",
        help="Which projector kernels to run on; leap needs benchmark/ on PYTHONPATH and LEAP installed",
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "cuda"),
        default="torch",
        help="torchtomo kernels: torch runs anywhere; cuda compiles CUDA kernels at first use with PyTorch's NVRTC",
    )
    parser.add_argument(
        "--geometry",
        choices=("parallel", "fan"),
        default="parallel",
        help="Beam geometry; fan uses FanBeam size-aware defaults unless --src-dist / --det-dist / --n-det are set",
    )
    parser.add_argument("--src-dist", type=float, default=None, help="Fan-beam source-to-isocentre distance in pixels")
    parser.add_argument(
        "--det-dist", type=float, default=None, help="Fan-beam isocentre-to-detector distance in pixels"
    )
    parser.add_argument("--n-det", type=int, default=None, help="Fan-beam detector bins")
    parser.add_argument(
        "--resume", action="store_true", help="Resume saved data, weights, optimizer, and epoch schedule"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory of packed CT slices from pack_ct_subset.py; omit to generate ellipse phantoms",
    )
    parser.add_argument("--lpd-iterations", type=int, default=5, help="Unrolled primal-dual iterations")
    parser.add_argument("--lpd-memory", type=int, default=5, help="Primal and dual memory channels")
    parser.add_argument("--lpd-width", type=int, default=24, help="Channels in each LPD update block")
    parser.add_argument("--unet-width", type=int, default=16, help="Channels at the U-Net's finest level")
    parser.add_argument("--iradon-width", type=int, default=16, help="Channels in the iRadonMAP refinement network")
    parser.add_argument(
        "--iradon-learning-rate",
        type=float,
        default=1e-4,
        help="iRadonMAP needs a smaller step than the other networks; 1e-3 tears its geometric layers apart",
    )
    parser.add_argument(
        "--n2i-splits", type=int, default=3, help="View subsets for Noise2Inverse; must divide --angles"
    )
    parser.add_argument("--p2p-grid", type=int, default=4, help="Proj2Proj mask grid side in the projection domain")
    parser.add_argument(
        "--models",
        default="fbp-unet,lpd,iradonmap,noise2inverse,proj2proj",
        help="Comma separated list of the networks to train and score",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true", help="Reload saved data and best weights without retraining"
    )
    parser.add_argument("--sirt-iterations", type=int, default=200, help="Longest SIRT run offered to the search")
    parser.add_argument("--sirt-relaxation", type=float, default=1.0)
    parser.add_argument("--sart-sweeps", type=int, default=20, help="Longest SART run offered to the search")
    parser.add_argument("--sart-relaxations", default="1,0.3,0.1,0.03", help="SART relaxations searched on validation")
    parser.add_argument(
        "--sart-subsets", type=int, default=0, help="Angle subsets per SART sweep; 0 uses one view at a time"
    )
    parser.add_argument("--red-iterations", type=int, default=100, help="Longest RED run offered to the search")
    parser.add_argument("--red-step", type=float, default=1.0)
    parser.add_argument("--red-weights", default="0.1,0.3,1,3,10", help="RED prior weights searched on validation")
    parser.add_argument(
        "--bm3d-sigmas", default="0.01,0.02,0.05,0.1,0.2", help="BM3D noise levels searched on validation"
    )
    parser.add_argument("--skip-classical", action="store_true", help="Evaluate only FBP and the two trained networks")
    parser.add_argument(
        "--grid-tile", type=int, default=256, help="Largest panel size in the PNG figures; 0 keeps full resolution"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    setup_logging(output)
    evaluate_only = args.evaluate_only
    if evaluate_only and args.resume:
        parser.error("choose either --evaluate-only or --resume")
    if evaluate_only or args.resume:
        config = json.loads((output / "config.json").read_text())
        # Keep the caller's device, threads, and output location.
        for key in (
            "image_size",
            "angles",
            "batch_size",
            "seed",
            "unet_epochs",
            "lpd_epochs",
            "learning_rate",
            "target_psnr",
            "data_dir",
            "lpd_iterations",
            "lpd_memory",
            "lpd_width",
            "unet_width",
            "iradon_width",
            "iradon_learning_rate",
            "n2i_splits",
            "p2p_grid",
            "models",
            "projector",
            "geometry",
            "src_dist",
            "det_dist",
            "n_det",
        ):
            if key in config:
                setattr(args, key, config[key])
            else:
                # Result directories written before an option existed keep the current default.
                config[key] = getattr(args, key)
                LOGGER.warning("config.json has no %s; using %r", key, config[key])
    sizes = (
        args.image_size,
        args.angles,
        args.unet_epochs,
        args.lpd_epochs,
        args.batch_size,
        args.threads,
        args.lpd_iterations,
        args.lpd_memory,
        args.lpd_width,
        args.unet_width,
    )
    if min(sizes) < 1:
        parser.error("sizes, epochs, batch size, thread count, and model widths must be positive")
    if args.micro_batch < 0:
        parser.error("micro batch size cannot be negative")
    if args.lpd_memory < 2:
        parser.error("LPD needs at least two memory channels; the forward operator reads the second one")
    if args.image_size < 8 or args.image_size % 4:
        parser.error("image size must be at least 8 and divisible by 4")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    LOGGER.info(
        "run device=%s torch=%s geometry=%s image_size=%d angles=%d seed=%d",
        args.device,
        torch.__version__,
        getattr(args, "geometry", "parallel"),
        args.image_size,
        args.angles,
        args.seed,
    )
    if evaluate_only or args.resume:
        data = torch.load(output / "dataset.pt", map_location="cpu")
        projector = build_projector(args)
        operator_norm = config["operator_norm"]
        summary_path = output / "training-summary.json"
        training = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    else:
        projector, data, photons, _ = prepare_data(args, output)
        operator_norm = estimate_operator_norm(projector)
        config = {
            **{
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
                if key not in ("output", "evaluate_only", "resume")
            },
            "photons": photons,
            "operator_norm": operator_norm,
            "noise_model": "Poisson(I0 * exp(-Ax)), then -log(max(counts, 1) / I0)",
            "metric": "mean per-image PSNR; full image; data_range=1; no prediction clipping",
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "cuda_device": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "cuda_version": torch.version.cuda if args.device == "cuda" else None,
        }
        write_json(output / "config.json", config)
        LOGGER.info("operator_norm=%.8f", operator_norm)
        training = {}

    # Objectives reconstruct through the projector, so it has to be in place first.
    projector.to(args.device)

    def build(name):
        """Each method as (model, objective, epochs, learning rate), with its own seed.

        iRadonMAP needs a smaller step than the rest. Its filtering and back-projection
        layers start at the analytic reconstruction, and an Adam step of 1e-3 moves each
        back-projection weight by a few percent of its initial value, which pulls the
        geometry apart faster than the refinement network can make use of it.
        """
        offsets = {"fbp-unet": 4, "lpd": 5, "iradonmap": 7, "noise2inverse": 8, "proj2proj": 9}
        torch.manual_seed(args.seed + offsets[name])
        unet_of = lambda: FBPUNet(projector.circle_mask, width=config["unet_width"])  # noqa: E731
        if name == "fbp-unet":
            return unet_of(), Supervised(data["fbp"], data["truth"]), args.unet_epochs, args.learning_rate
        if name == "lpd":
            model = LearnedPrimalDual(
                projector,
                operator_norm,
                iterations=config["lpd_iterations"],
                memory=config["lpd_memory"],
                width=config["lpd_width"],
            )
            return model, Supervised(data["noisy"], data["truth"]), args.lpd_epochs, args.learning_rate
        if name == "iradonmap":
            model = IRadonMap(projector, width=config["iradon_width"])
            return model, Supervised(data["noisy"], data["truth"]), args.unet_epochs, args.iradon_learning_rate
        if name == "noise2inverse":
            objective = Noise2Inverse(projector, data["noisy"], config["n2i_splits"])
            return unet_of(), objective, args.unet_epochs, args.learning_rate
        if name == "proj2proj":
            objective = Proj2Proj(projector, data["noisy"], config["p2p_grid"])
            return unet_of(), objective, args.unet_epochs, args.learning_rate
        raise ValueError(f"unknown model {name}")

    selected = [name for name in config["models"].split(",") if name]
    model_specs = [(name, *build(name)) for name in selected]
    for name, model, objective, epochs, learning_rate in model_specs:
        if evaluate_only:
            load_weights(model, torch.load(output / f"{name}-best.pt", map_location="cpu")["state_dict"])
            model.to(args.device)
        else:
            training[name] = train_model(
                name, model, objective, data, args, output, epochs, args.seed + 6, learning_rate
            )
            write_json(output / "training-summary.json", training)

    test_ids = data["splits"]["test"]
    region = projector.circle_mask.bool()
    _, fbp_seconds = fbp_split_seconds(projector, data["noisy"][test_ids], args.device, args.batch_size)
    scored = {"fbp": data["fbp"][test_ids]}
    for name, model, objective, _, _ in model_specs:
        scored[name] = objective.reconstruct(model, test_ids, args.device, args.batch_size)
    selection = {}
    if not args.skip_classical:
        denoiser = dict((name, model) for name, model, *_ in model_specs).get("fbp-unet")
        untrained, selection = run_classical(projector, data, denoiser, operator_norm, args, region)
        write_json(output / "classical-selection.json", selection)
        scored.update(untrained)
    # Analytic, then untrained iterative, then learned, with RED beside the network it reuses.
    order = (
        "fbp",
        "sirt",
        "sart",
        "bm3d",
        "fbp-unet",
        "red",
        "iradonmap",
        "noise2inverse",
        "proj2proj",
        "lpd",
    )
    reconstructions = {name: scored[name] for name in order if name in scored}
    test_windows = data["windows"][test_ids] if "windows" in data else None
    metrics = {}
    for name, prediction in reconstructions.items():
        metrics[name] = summarize(prediction, data["truth"][test_ids], region=region, windows=test_windows)
    record_method_seconds(metrics, training, selection, fbp_seconds)
    for name in metrics:
        LOGGER.info(
            "TEST method=%s mean_psnr_db=%.4f roi_psnr_db=%.4f window_psnr_db=%s "
            "std_psnr_db=%.4f mse=%.7f seconds=%.2f",
            name,
            metrics[name]["mean_psnr_db"],
            metrics[name]["mean_roi_psnr_db"],
            f"{metrics[name]['mean_window_psnr_db']:.4f}" if test_windows is not None else "n/a",
            metrics[name]["std_psnr_db"],
            metrics[name]["mse"],
            metrics[name]["seconds"],
        )
    # Recorded as a reference rather than a method: what the analytic reconstruction
    # would give from the same geometry with no noise at all.
    with torch.no_grad():
        noiseless = torch.cat(
            [projector.fbp(batch.to(args.device)).cpu() for batch in data["clean"][test_ids].split(args.batch_size)]
        )
    reference = summarize(noiseless, data["truth"][test_ids], region=region, windows=test_windows)
    LOGGER.info(
        "TEST reference=noiseless-fbp mean_psnr_db=%.4f roi_psnr_db=%.4f window_psnr_db=%s",
        reference["mean_psnr_db"],
        reference["mean_roi_psnr_db"],
        f"{reference['mean_window_psnr_db']:.4f}" if test_windows is not None else "n/a",
    )
    write_json(
        output / "metrics.json",
        {"split": "test", "ids": test_ids.tolist(), "methods": metrics, "noiseless_fbp_reference": reference},
    )
    torch.save({"ids": test_ids, "truth": data["truth"][test_ids], **reconstructions}, output / "reconstructions.pt")
    supervision = {name: objective.uses_truth for name, _, objective, *_ in model_specs}
    save_plots(output, data, reconstructions, metrics, supervision, tile=args.grid_tile)
    LOGGER.info("complete outputs=%s", output)


if __name__ == "__main__":
    main()
