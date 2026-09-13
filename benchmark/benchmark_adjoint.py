"""Compare the discrete adjoint with the historical analytical backprojection.

Run from the repository root:
    PYTHONPATH=src python benchmark/benchmark_adjoint.py --pairs 500 --dtype float64

Per-pair error: |lhs-rhs| / max(|lhs|, |rhs|, tiny).
The fitted alpha minimizes ||lhs - alpha * rhs||_2 over all sampled pairs.
The norm defect uses ||Ax|| ||y|| + ||x|| ||A^T y|| to avoid cancellation.
"""

import argparse
import json

import torch

from torchtomo import FanBeam, ParallelBeam


def _summary(errors):
    return {
        "mean": errors.mean().item(),
        "median": errors.quantile(0.5).item(),
        "std": errors.std(unbiased=False).item(),
        "p95": errors.quantile(0.95).item(),
        "max": errors.max().item(),
    }


def _relative_error(lhs, rhs):
    return (lhs - rhs).abs() / torch.maximum(lhs.abs(), rhs.abs()).clamp_min(torch.finfo(lhs.dtype).tiny)


def measure(projector, pairs, batch_size, seed):
    """Sample independent standard-normal image/sinogram pairs on CPU."""
    generator = torch.Generator().manual_seed(seed)
    dtype = projector.angles.dtype
    products, defects = [], []
    for start in range(0, pairs, batch_size):
        count = min(batch_size, pairs - start)
        x = torch.randn(count, 1, projector.img_size, projector.img_size, dtype=dtype, generator=generator)
        y = torch.randn(count, 1, projector.n_angles, projector.n_det, dtype=dtype, generator=generator)
        with torch.no_grad():
            ax = projector(x)
            by = projector.backproject(y)
            aty = projector.backward(y)
            # Compute dots in the selected dtype; aggregate the metrics in float64.
            lhs = (ax * y).flatten(1).sum(1).double()
            old_rhs = (x * by).flatten(1).sum(1).double()
            rhs = (x * aty).flatten(1).sum(1).double()
            scale = ax.flatten(1).norm(dim=1) * y.flatten(1).norm(dim=1)
            scale += x.flatten(1).norm(dim=1) * aty.flatten(1).norm(dim=1)
            defects.append((lhs - rhs).abs() / scale.double().clamp_min(torch.finfo(dtype).tiny))
            products.append(torch.stack((lhs, old_rhs, rhs), dim=1))
    lhs, old_rhs, rhs = torch.cat(products).unbind(1)
    alpha = torch.dot(lhs, old_rhs) / torch.dot(old_rhs, old_rhs)
    return {
        "geometry": repr(projector),
        "analytical_backproject_relative_error": _summary(_relative_error(lhs, old_rhs)),
        "scale_alpha_applied_to_backproject": alpha.item(),
        "scaled_backproject_relative_error": _summary(_relative_error(lhs, alpha * old_rhs)),
        "scaled_backproject_inner_product_ls_residual": ((lhs - alpha * old_rhs).norm() / lhs.norm()).item(),
        "exact_backward_relative_error": _summary(_relative_error(lhs, rhs)),
        "exact_backward_inner_product_ls_residual": ((lhs - rhs).norm() / lhs.norm()).item(),
        "exact_backward_max_norm_defect": torch.cat(defects).max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--n-angles", type=int, default=45)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.pairs, args.batch_size, args.img_size, args.n_angles) < 1:
        parser.error("pair count, batch size, image size, and angle count must be positive")
    torch.set_num_threads(2)
    projectors = [
        ParallelBeam(img_size=args.img_size, n_angles=args.n_angles),
        FanBeam(
            img_size=args.img_size,
            n_angles=args.n_angles,
            n_det=3 * args.img_size // 2,
            src_dist=2 * args.img_size,
            det_dist=2 * args.img_size,
            n_samples=64,
        ),
    ]
    results = [
        measure(projector.to(dtype=getattr(torch, args.dtype)), args.pairs, args.batch_size, args.seed)
        for projector in projectors
    ]
    print(json.dumps({"torch_version": torch.__version__, "device": "cpu", **vars(args), "results": results}, indent=2))


if __name__ == "__main__":
    main()
