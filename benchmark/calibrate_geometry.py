"""Recover a centre-of-rotation error by differentiating through the geometry.

Run from the repository root:
    PYTHONPATH=src python benchmark/calibrate_geometry.py

The scanner's axis of rotation is off centre by a few pixels and the
reconstruction does not know it. The only thing measured is the sinogram, so the
objective is the reconstruction's own data consistency,

    L(u) = || A_u fbp_u(y) - y ||^2 / N,

with the detector shift u as the single free parameter. Nothing here knows the
phantom: the gradient dL/du comes from the projector itself, which is what no
other CT library offers.

The pose table is assembled as an expression in u, one scalar driving every view,
so the gradient arrives at u through ordinary autograd. A per-view shift, a
subset of the angles, or a rigid body model would be the same code with a
different expression.
"""

import argparse
import time

import torch

from torchtomo import FanBeam, ParallelBeam, shepp_logan


def _build(name: str, size: int, n_angles: int, device: torch.device):
    kind = ParallelBeam if name == "parallel" else FanBeam
    return kind(img_size=size, n_angles=n_angles, circle=True).to(device)


def _pose_with_shift(base: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """The pose table as an expression in one scalar: same angles, shifted detector."""
    columns = [base[:, 0], base[:, 1] + shift]
    columns += [base[:, index] for index in range(2, base.shape[1])]
    return torch.stack(columns, dim=1)


def measure(sinogram, projector, base, shift):
    projector.pose = _pose_with_shift(base, shift)
    return (projector.forward(projector.fbp(sinogram)) - sinogram).pow(2).mean()


def calibrate(name, size, n_angles, truth_shift, steps, lr, noise, seed, device):
    torch.manual_seed(seed)
    phantom = shepp_logan(size).view(1, 1, size, size).to(device)

    scanner = _build(name, size, n_angles, device)
    scanner.set_pose(detector_shift=truth_shift)
    with torch.no_grad():
        sinogram = scanner.forward(phantom)
        if noise > 0:
            sinogram = sinogram + noise * sinogram.std() * torch.randn_like(sinogram)

    projector = _build(name, size, n_angles, device)
    base = projector.pose.detach().clone()
    shift = torch.zeros((), dtype=base.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([shift], lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr / 500)

    history = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = measure(sinogram, projector, base, shift)
        loss.backward()
        optimizer.step()
        schedule.step()
        history.append((step, loss.item(), shift.item()))
    with torch.no_grad():
        final = measure(sinogram, projector, base, shift).item()
    history.append((steps, final, shift.item()))
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry", default="parallel", choices=("parallel", "fan", "both"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--angles", type=int, default=256)
    parser.add_argument("--shift", type=float, default=3.0, help="the scanner's axis offset in pixels")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--noise", type=float, default=0.0, help="Gaussian noise, relative to the sinogram's own std")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    names = ("parallel", "fan") if args.geometry == "both" else (args.geometry,)
    for name in names:
        print(f"\n{name} beam, {args.size} px, {args.angles} angles, axis off by {args.shift:.2f} px on {device}")
        start = time.perf_counter()
        history = calibrate(
            name, args.size, args.angles, args.shift, args.steps, args.lr, args.noise, args.seed, device
        )
        elapsed = time.perf_counter() - start
        print(f"{'step':>5}  {'loss':>12}  {'shift px':>9}  {'error px':>9}")
        for step, loss, shift in history:
            if step % max(1, args.steps // 10) == 0 or step == args.steps:
                print(f"{step:5d}  {loss:12.6e}  {shift:+9.4f}  {abs(shift - args.shift):9.4f}")
        error = abs(history[-1][2] - args.shift)
        print(
            f"recovered {history[-1][2]:.4f} px of a {args.shift:.2f} px offset, error {error:.4f} px, {elapsed:.1f} s"
        )


if __name__ == "__main__":
    main()
