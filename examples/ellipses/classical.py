"""SIRT, SART, and RED reconstructions, none of which are trained here.

SIRT and SART are the classical algebraic methods, driven by the same matched
operator pair the Learned Primal-Dual network uses. RED plugs the already
trained FBP U-Net in as the denoiser, so the same learned prior can be compared
with and without an explicit data consistency term.

All three take the noisy sinogram, project onto the non-negative orthant, and
stay inside the circle the geometry can see. Their iteration counts are chosen
on the validation split, never on the test split.
"""

import logging
import math
import time

import torch
from bm3d import denoise as bm3d_denoise
from data import apply_window, psnr_per_image

LOGGER = logging.getLogger("ellipses")


def reciprocal_or_zero(weights, relative_floor=1e-6):
    """Invert projection weights, leaving unreachable rays and pixels at zero."""
    floor = relative_floor * weights.max()
    return torch.where(weights > floor, 1 / weights.clamp_min(floor), torch.zeros_like(weights))


@torch.no_grad()
def projection_weights(projector):
    """Inverse row and column sums of A, the SIRT and SART preconditioners."""
    device = projector.angles.device
    ones_image = torch.ones(1, 1, projector.img_size, projector.img_size, device=device)
    ones_sinogram = torch.ones(1, 1, projector.n_angles, projector.n_det, device=device)
    return reciprocal_or_zero(projector.forward(ones_image)), reciprocal_or_zero(projector.adjoint(ones_sinogram))


def spread_order(count):
    """Visit subsets in a widely separated cycle rather than in angular order.

    Consecutive updates at neighbouring angles reinforce each other's streaks. A
    stride coprime with the subset count still touches every subset exactly once
    per sweep while jumping roughly across the arc each time.
    """
    stride = max(1, round(count / ((1 + 5**0.5) / 2)))
    while math.gcd(stride, count) != 1:
        stride -= 1
    return [(index * stride) % count for index in range(count)]


def subset_projectors(projector, count):
    """Split the angles into interleaved sub-projectors that each span the full arc.

    Taking every count-th angle keeps the subsets evenly spaced, which is what
    lets each one be rebuilt as an ordinary projector over its own angle range.
    The angular weight is rescaled so that a subset fbp() keeps the same
    normalisation as the parent's, which matters when subset reconstructions are
    compared with each other or with the full one.
    """
    if not 1 <= count <= projector.n_angles:
        raise ValueError(f"subset count {count} outside 1..{projector.n_angles}")
    subsets = []
    for offset in range(count):
        indices = torch.arange(offset, projector.n_angles, count)
        chosen = projector.angles[indices]
        # type(projector), not ParallelBeam, so a run on another backend keeps it.
        kwargs = dict(
            img_size=projector.img_size,
            n_angles=len(chosen),
            n_det=projector.n_det,
            circle=projector.circle,
            angles=chosen,
        )
        if hasattr(projector, "src_dist"):
            kwargs.update(
                src_dist=projector.src_dist,
                det_dist=projector.det_dist,
                det_width=projector.det_width,
                n_samples=projector.n_samples,
            )
        if getattr(projector, "backend", "torch") != "torch":
            kwargs["backend"] = projector.backend
        subset = type(projector)(**kwargs).to(projector.angles.device)
        if not torch.allclose(subset.angles, chosen, atol=1e-5):
            raise ValueError("angle subset is not evenly spaced; SART needs a uniform angle grid")
        subset.angle_step = projector.angle_step * projector.n_angles / len(chosen)
        subsets.append((indices, subset))
    return subsets


def angle_subsets(projector, count):
    """Sub-projectors with their SIRT weights, ordered for SART's update sweep."""
    blocks = [(indices, subset, projection_weights(subset)) for indices, subset in subset_projectors(projector, count)]
    return [blocks[position] for position in spread_order(count)]


def constrain(image, mask):
    return image.clamp_min(0) * mask


@torch.no_grad()
def sirt(projector, sinogram, iterations, weights, relaxation=1.0, callback=None):
    """x <- x + w * C A^T R (y - A x), starting from zero."""
    row, column = weights
    mask = projector.circle_mask[None, None]
    image = sinogram.new_zeros(sinogram.shape[0], 1, projector.img_size, projector.img_size)
    for step in range(1, iterations + 1):
        residual = (sinogram - projector.forward(image)) * row
        image = constrain(image + relaxation * column * projector.adjoint(residual), mask)
        if callback is not None:
            callback(step, image)
    return image


@torch.no_grad()
def sart(projector, sinogram, updates, blocks, relaxation=1.0, callback=None):
    """The same update as SIRT, applied to one subset of views at a time.

    The budget counts subset updates rather than whole sweeps, because with one
    view per subset a sweep is as wide as the angle count and the best stopping
    point usually falls inside one.
    """
    mask = projector.circle_mask[None, None]
    image = sinogram.new_zeros(sinogram.shape[0], 1, projector.img_size, projector.img_size)
    for update in range(1, updates + 1):
        indices, subset, (row, column) = blocks[(update - 1) % len(blocks)]
        residual = (sinogram[:, :, indices] - subset.forward(image)) * row
        image = constrain(image + relaxation * column * subset.adjoint(residual), mask)
        if callback is not None:
            callback(update, image)
    return image


@torch.no_grad()
def red(projector, sinogram, initial, denoiser, iterations, weight, operator_norm, step=1.0, callback=None):
    """Steepest descent on ||A x - y||^2 / (2 ||A||^2) + weight * x'(x - D(x)) / 2, from the FBP image.

    Dividing the data term by the squared operator norm fixes its Lipschitz
    constant at one, and dividing the whole step by 1 + weight keeps it stable
    across the weight sweep: weight 0 is plain least squares and a large weight
    approaches the denoiser's own fixed point.
    """
    mask = projector.circle_mask[None, None]
    image = initial.clone()
    for iteration in range(1, iterations + 1):
        data_gradient = projector.adjoint(projector.forward(image) - sinogram) / operator_norm**2
        gradient = data_gradient + weight * (image - denoiser(image))
        image = constrain(image - step * gradient / (1 + weight), mask)
        if not torch.isfinite(image).all():
            raise FloatingPointError(f"RED diverged at iteration {iteration} with weight {weight}")
        if callback is not None:
            callback(iteration, image)
    return image


@torch.no_grad()
def score_curve(reconstruct, sinograms, truth, windows, region, device, batch_size, extra=()):
    """Mean PSNR after every iteration, without keeping any iterate around."""
    totals = {}
    batches = zip(
        sinograms.split(batch_size),
        truth.split(batch_size),
        zip(*[item.split(batch_size) for item in extra]) if extra else [()] * len(sinograms.split(batch_size)),
        windows.split(batch_size) if windows is not None else [None] * len(sinograms.split(batch_size)),
    )
    for sinogram, target, rest, window in batches:
        target = target.to(device)
        window = None if window is None else window.to(device)

        def record(step, image, target=target, window=window):
            if window is None:
                scores = psnr_per_image(image, target, region=region)
            else:
                scores = psnr_per_image(apply_window(image, window), apply_window(target, window), region=region)
            totals[step] = totals.get(step, 0.0) + scores.sum().item()

        reconstruct(sinogram.to(device), [item.to(device) for item in rest], record)
    return [{"iteration": step, "psnr_db": total / len(sinograms)} for step, total in sorted(totals.items())]


@torch.no_grad()
def reconstruct_split(reconstruct, sinograms, device, batch_size, extra=()):
    rests = zip(*[item.split(batch_size) for item in extra]) if extra else [()] * len(sinograms.split(batch_size))
    outputs = [
        reconstruct(sinogram.to(device), [item.to(device) for item in rest], None).cpu()
        for sinogram, rest in zip(sinograms.split(batch_size), rests)
    ]
    return torch.cat(outputs)


def method_table(projector, unet, operator_norm, args):
    """Every method as (inputs, unit, settings, max steps, factory, extra record)."""
    weights = projection_weights(projector)
    blocks = angle_subsets(projector, args.sart_subsets or projector.n_angles)
    LOGGER.info(
        "classical sart_subsets=%d sirt_iterations=%d sart_sweeps=%d red_iterations=%d",
        len(blocks),
        args.sirt_iterations,
        args.sart_sweeps,
        args.red_iterations,
    )

    def sirt_factory(setting, steps):
        return lambda sinogram, extra, callback: sirt(
            projector, sinogram, steps, weights, args.sirt_relaxation, callback
        )

    def sart_factory(setting, steps):
        return lambda sinogram, extra, callback: sart(projector, sinogram, steps, blocks, setting, callback)

    def bm3d_factory(setting, steps):
        # The input here is the FBP image, not a sinogram: BM3D is a post-processor.
        def run(image, extra, callback):
            result = constrain(bm3d_denoise(image, setting), projector.circle_mask[None, None])
            if callback is not None:
                callback(1, result)
            return result

        return run

    def red_factory(setting, steps):
        return lambda sinogram, extra, callback: red(
            projector, sinogram, extra[0], unet, steps, setting, operator_norm, args.red_step, callback
        )

    table = {
        "sirt": (
            ("noisy",),
            "iterations",
            [None],
            args.sirt_iterations,
            sirt_factory,
            {"relaxation": args.sirt_relaxation},
        ),
        "bm3d": (
            ("fbp",),
            "stages",
            [float(value) for value in args.bm3d_sigmas.split(",")],
            1,
            bm3d_factory,
            {"setting_name": "sigma"},
        ),
        "sart": (
            ("noisy",),
            "subset updates",
            [float(value) for value in args.sart_relaxations.split(",")],
            args.sart_sweeps * len(blocks),
            sart_factory,
            {"subsets_per_sweep": len(blocks), "setting_name": "relaxation"},
        ),
    }
    # RED borrows the trained U-Net, so it only exists when that network was built.
    if unet is not None:
        table["red"] = (
            ("noisy", "fbp"),
            "iterations",
            [float(value) for value in args.red_weights.split(",")],
            args.red_iterations,
            red_factory,
            {"step": args.red_step, "setting_name": "prior weight"},
        )
    return table


def run_classical(projector, data, unet, operator_norm, args, region):
    """Tune each method on validation, then reconstruct the test split once."""
    device = torch.device(args.device)
    val_ids, test_ids = data["splits"]["val"], data["splits"]["test"]
    windows = data.get("windows")
    if unet is not None:
        unet.eval()
    reconstructions, selection = {}, {}
    table = method_table(projector, unet, operator_norm, args)
    for name, (inputs, unit, settings, max_steps, factory, notes) in table.items():
        primary, extra_keys = inputs[0], inputs[1:]
        curves, best = {}, None
        started = time.perf_counter()
        for setting in settings:
            curve = score_curve(
                factory(setting, max_steps),
                data[primary][val_ids],
                data["truth"][val_ids],
                None if windows is None else windows[val_ids],
                region,
                device,
                args.batch_size,
                extra=tuple(data[key][val_ids] for key in extra_keys),
            )
            label = "default" if setting is None else f"{setting:g}"
            curves[label] = curve
            top = max(curve, key=lambda row: row["psnr_db"])
            LOGGER.info(
                "classical method=%s setting=%s best_val_%s=%d val_psnr_db=%.4f",
                name,
                label,
                unit,
                top["iteration"],
                top["psnr_db"],
            )
            if best is None or top["psnr_db"] > best[2]:
                best = (setting, top["iteration"], top["psnr_db"], label)
        setting, steps, score, label = best
        reconstructions[name] = reconstruct_split(
            factory(setting, steps),
            data[primary][test_ids],
            device,
            args.batch_size,
            extra=tuple(data[key][test_ids] for key in extra_keys),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        selection[name] = {
            "unit": unit,
            "selected_steps": steps,
            "selected_setting": setting,
            "val_psnr_db": score,
            "searched_steps": max_steps,
            "seconds": seconds,
            **notes,
            "curves": curves,
        }
        LOGGER.info(
            "classical method=%s selected %s=%d setting=%s val_psnr_db=%.4f seconds=%.2f",
            name,
            unit,
            steps,
            label,
            score,
            seconds,
        )
    return reconstructions, selection
