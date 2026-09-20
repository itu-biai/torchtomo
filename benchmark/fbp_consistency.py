"""Is a library's FBP the inverse of its own forward projector?

Reprojecting a reconstruction should return the measurements it came from. This
reports the least squares gain between the two and the relative residual left
after reprojection, plus the constant offset the reconstruction carries against
the phantom, for torchtomo and, when it is installed, LEAP.

    PYTHONPATH=src:benchmark python benchmark/fbp_consistency.py --size 512

LEAP is an optional benchmark dependency. Nothing in the library imports it.
"""

import argparse

import torch

from torchtomo import ParallelBeam, shepp_logan

try:
    from leap_projector import LeapParallelBeam
except ImportError:
    LeapParallelBeam = None


def measure(projector, phantom):
    """Gain, residual, and reconstruction bias for one projector."""
    sinogram = projector.forward(phantom)
    recon = projector.fbp(sinogram)
    reprojected = projector.forward(recon)
    gain = (reprojected * sinogram).sum() / sinogram.square().sum()
    residual = (reprojected - sinogram).norm() / sinogram.norm()
    mask = projector.circle_mask.bool()
    bias = (recon - phantom)[:, :, mask].mean()
    return float(gain), float(residual), float(bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--angles", type=int, nargs="+", default=[45, 90, 180, 360, 720])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    phantom = shepp_logan(size=args.size, device=device)
    libraries = [("torchtomo", ParallelBeam)]
    if LeapParallelBeam is not None and device.type == "cuda":
        libraries.append(("leap", LeapParallelBeam))

    print(f"{args.size} px, Shepp-Logan, gain / residual / bias")
    header = "| Angles | " + " | ".join(name for name, _ in libraries) + " |"
    print(header)
    print("| ---: |" + " ---: |" * len(libraries))
    for n_angles in args.angles:
        cells = []
        for _, cls in libraries:
            projector = cls(img_size=args.size, n_angles=n_angles).to(device)
            gain, residual, bias = measure(projector, phantom)
            cells.append(f"{gain:.4f} / {residual:.4f} / {bias:+.5f}")
        print(f"| {n_angles} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
