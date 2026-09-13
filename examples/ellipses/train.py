"""Train and compare FBP, FBP+U-Net, and LPD on 100 synthetic ellipse phantoms.

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
from data import apply_in_batches, calibrate_photons, make_phantoms, make_splits, poisson_sinogram, psnr_per_image
from models import FBPUNet, LearnedPrimalDual, estimate_operator_norm
from torch.nn import functional as F

from torchtomo import ParallelBeam

LOGGER = logging.getLogger("ellipses")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def setup_logging(output):
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output / "training.log", mode="a")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def prepare_data(args, output):
    projector = ParallelBeam(img_size=args.image_size, n_angles=args.angles)
    truth, ellipses = make_phantoms(size=args.image_size, seed=args.seed)
    splits = make_splits(args.seed + 1)
    assert [len(splits[key]) for key in ("train", "val", "test")] == [60, 20, 20]
    assert len(set(torch.cat(list(splits.values())).tolist())) == 100
    clean = apply_in_batches(projector.forward, truth, args.batch_size)
    clean_fbp = apply_in_batches(projector.fbp, clean, args.batch_size)
    LOGGER.info(
        "clean train FBP PSNR=%.4f dB", psnr_per_image(clean_fbp[splits["train"]], truth[splits["train"]]).mean()
    )
    photons, calibration = calibrate_photons(
        projector, clean[splits["train"]], truth[splits["train"]], target=args.target_psnr, seed=args.seed + 2
    )
    # A fresh fixed realization is shared by all three methods, after calibration.
    noisy, counts = poisson_sinogram(clean, photons, args.seed + 3)
    fbp = apply_in_batches(projector.fbp, noisy, args.batch_size)
    data = {"truth": truth, "clean": clean, "noisy": noisy, "counts": counts, "fbp": fbp, "splits": splits}
    torch.save(data, output / "dataset.pt")
    write_json(output / "splits.json", {key: value.tolist() for key, value in splits.items()})
    write_json(
        output / "phantoms.json",
        {"parameter_order": ["intensity", "a", "b", "cx", "cy", "radians"], "ellipses": ellipses},
    )
    write_json(
        output / "noise-calibration.json", {"split": "train", "trials": calibration, "selected_photons": photons}
    )
    LOGGER.info(
        "dataset size=100 split=60/20/20 photons=%.3f zero_count_fraction=%.8f", photons, (counts == 0).float().mean()
    )
    # Do not evaluate the test split until after model/checkpoint selection.
    for name in ("train", "val"):
        ids = splits[name]
        LOGGER.info("baseline split=%s fbp_psnr_db=%.4f", name, psnr_per_image(fbp[ids], truth[ids]).mean())
    return projector, data, photons


@torch.no_grad()
def predict(model, inputs, device, batch_size):
    model.eval()
    return torch.cat([model(batch.to(device)).cpu() for batch in inputs.split(batch_size)])


def summarize(prediction, truth):
    scores = psnr_per_image(prediction, truth)
    return {
        "mse": F.mse_loss(prediction, truth).item(),
        "mean_psnr_db": scores.mean().item(),
        "std_psnr_db": scores.std(unbiased=False).item(),
        "per_image_psnr_db": scores.tolist(),
    }


def train_model(name, model, inputs, data, args, output, epochs, seed):
    device = torch.device(args.device)
    model.to(device)
    train_ids, val_ids = data["splits"]["train"], data["splits"]["val"]
    truth = data["truth"]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.learning_rate * 0.01)
    generator = torch.Generator().manual_seed(seed)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    LOGGER.info("model=%s parameters=%d epochs=%d device=%s", name, parameter_count, epochs, device)
    history, best_mse, best_epoch = [], float("inf"), 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = train_ids[torch.randperm(len(train_ids), generator=generator)]
        squared_error = 0.0
        lr = optimizer.param_groups[0]["lr"]
        for step, ids in enumerate(order.split(args.batch_size)):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs[ids].to(device))
            loss = F.mse_loss(prediction, truth[ids].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {name} loss at epoch {epoch}")
            loss.backward()
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
            squared_error += loss.item() * len(ids)
        validation = predict(model, inputs[val_ids], device, args.batch_size)
        val = summarize(validation, truth[val_ids])
        record = {
            "epoch": epoch,
            "train_mse": squared_error / len(train_ids),
            "val_mse": val["mse"],
            "val_psnr_db": val["mean_psnr_db"],
            "learning_rate": lr,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        if val["mse"] < best_mse:
            best_mse, best_epoch = val["mse"], epoch
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val_mse": best_mse},
                output / f"{name}-best.pt",
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
    checkpoint = torch.load(output / f"{name}-best.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    duration = time.perf_counter() - start
    LOGGER.info("model=%s training_complete seconds=%.2f best_epoch=%d", name, duration, best_epoch)
    return {"parameters": parameter_count, "best_epoch": best_epoch, "epochs": epochs, "training_seconds": duration}


def save_plots(output, data, reconstructions, metrics):
    # Matplotlib is already part of the repository's existing dev extra.
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_ids = data["splits"]["test"].tolist()
    truth = data["truth"][test_ids]
    images = {"gt": truth, **reconstructions}
    labels = {"gt": "Ground truth", "fbp": "FBP", "fbp-unet": "FBP + U-Net", "lpd": "Learned Primal-Dual"}
    # Every PNG has the same ordering and [0, 1] display range. Metrics use raw values.
    for name, tensor in images.items():
        tiles = tensor[:, 0].clamp(0, 1)
        rows = [torch.cat(list(tiles[start : start + 5]), dim=1) for start in range(0, 20, 5)]
        grid = torch.cat(rows, dim=0).repeat_interleave(3, 0).repeat_interleave(3, 1)
        plt.imsave(output / f"{name}.png", grid.numpy(), cmap="gray", vmin=0, vmax=1)

    # Compare the first four held-out images, selected by split order, not by score.
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), constrained_layout=True)
    for row in range(4):
        for column, (name, tensor) in enumerate(images.items()):
            ax = axes[row, column]
            ax.imshow(tensor[row, 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if name == "gt":
                subtitle = f"Phantom {test_ids[row]:03d}"
            else:
                score = metrics[name]["per_image_psnr_db"][row]
                subtitle = f"{score:.2f} dB"
            ax.set_title(f"{labels[name]}\n{subtitle}", fontsize=10)
            ax.set_axis_off()
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, color in (("fbp-unet", "#2764b0"), ("lpd", "#cf5b2e")):
        history = json.loads((output / f"{name}-history.json").read_text())
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_mse"] for row in history], "--", color=color, label=f"{labels[name]} train")
        axes[0].plot(epochs, [row["val_mse"] for row in history], color=color, label=f"{labels[name]} validation")
        axes[1].plot(epochs, [row["val_psnr_db"] for row in history], color=color, label=labels[name])
        best = min(history, key=lambda row: row["val_mse"])
        axes[0].scatter(best["epoch"], best["val_mse"], color=color, edgecolor="black", zorder=3)
        axes[1].scatter(
            best["epoch"],
            best["val_psnr_db"],
            color=color,
            edgecolor="black",
            zorder=3,
            label=f"Selected epoch {best['epoch']}",
        )
    val_ids = data["splits"]["val"]
    baseline = psnr_per_image(data["fbp"][val_ids], data["truth"][val_ids]).mean().item()
    axes[1].axhline(baseline, color="#777777", linestyle=":", label=f"FBP validation ({baseline:.2f} dB)")
    axes[0].set(ylabel="Mean squared error", yscale="log", title="Training and validation loss")
    axes[1].set(ylabel="Mean PSNR (dB)", title="Validation reconstruction quality")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "curves.png", dpi=150)
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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--evaluate-only", action="store_true", help="Reload saved data and best weights without retraining"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    setup_logging(output)
    evaluate_only = args.evaluate_only
    if evaluate_only:
        config = json.loads((output / "config.json").read_text())
        # Keep the caller's device, threads, and output location.
        for key in ("image_size", "angles", "batch_size", "seed", "unet_epochs", "lpd_epochs"):
            setattr(args, key, config[key])
    if min(args.image_size, args.angles, args.unet_epochs, args.lpd_epochs, args.batch_size, args.threads) < 1:
        parser.error("sizes, epochs, batch size, and thread count must be positive")
    if args.image_size < 8 or args.image_size % 4:
        parser.error("image size must be at least 8 and divisible by 4")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    LOGGER.info(
        "run device=%s torch=%s image_size=%d angles=%d seed=%d",
        args.device,
        torch.__version__,
        args.image_size,
        args.angles,
        args.seed,
    )
    if evaluate_only:
        data = torch.load(output / "dataset.pt", map_location="cpu")
        projector = ParallelBeam(img_size=args.image_size, n_angles=args.angles)
        operator_norm = config["operator_norm"]
        training = json.loads((output / "training-summary.json").read_text())
    else:
        projector, data, photons = prepare_data(args, output)
        operator_norm = estimate_operator_norm(projector)
        config = {
            **{key: value for key, value in vars(args).items() if key not in ("output", "evaluate_only")},
            "photons": photons,
            "operator_norm": operator_norm,
            "noise_model": "Poisson(I0 * exp(-Ax)), then -log(max(counts, 1) / I0)",
            "metric": "mean per-image PSNR; full image; data_range=1; no prediction clipping",
            "lpd_iterations": 5,
            "lpd_memory": 5,
            "lpd_width": 24,
            "unet_width": 16,
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
        }
        write_json(output / "config.json", config)
        LOGGER.info("operator_norm=%.8f", operator_norm)
        training = {}

    torch.manual_seed(args.seed + 4)
    unet = FBPUNet(projector.circle_mask, width=config["unet_width"])
    torch.manual_seed(args.seed + 5)
    lpd = LearnedPrimalDual(
        projector,
        operator_norm,
        iterations=config["lpd_iterations"],
        memory=config["lpd_memory"],
        width=config["lpd_width"],
    )
    model_specs = (("fbp-unet", unet, "fbp", args.unet_epochs), ("lpd", lpd, "noisy", args.lpd_epochs))
    for name, model, input_key, epochs in model_specs:
        if evaluate_only:
            model.load_state_dict(torch.load(output / f"{name}-best.pt", map_location="cpu")["state_dict"])
            model.to(args.device)
        else:
            training[name] = train_model(name, model, data[input_key], data, args, output, epochs, args.seed + 6)
            write_json(output / "training-summary.json", training)

    test_ids = data["splits"]["test"]
    reconstructions = {"fbp": data["fbp"][test_ids]}
    for name, model, input_key, _ in model_specs:
        reconstructions[name] = predict(model, data[input_key][test_ids], args.device, args.batch_size)
    metrics = {}
    for name, prediction in reconstructions.items():
        metrics[name] = summarize(prediction, data["truth"][test_ids])
        LOGGER.info(
            "TEST method=%s mean_psnr_db=%.4f std_psnr_db=%.4f mse=%.7f",
            name,
            metrics[name]["mean_psnr_db"],
            metrics[name]["std_psnr_db"],
            metrics[name]["mse"],
        )
    write_json(output / "metrics.json", {"split": "test", "ids": test_ids.tolist(), "methods": metrics})
    torch.save({"ids": test_ids, "truth": data["truth"][test_ids], **reconstructions}, output / "reconstructions.pt")
    save_plots(output, data, reconstructions, metrics)
    LOGGER.info("complete outputs=%s", output)


if __name__ == "__main__":
    main()
