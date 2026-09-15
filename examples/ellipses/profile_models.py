"""Measure full-resolution forward/backward time before starting a long run.

Named profile_models rather than profile so it cannot shadow the standard library
module that cProfile imports when torch loads its dynamo package.
"""

import argparse
import logging
import time

import torch
from models import FBPUNet, LearnedPrimalDual

from torchtomo import ParallelBeam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--angles", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable to this process")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to this process")
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    projector = ParallelBeam(img_size=args.image_size, n_angles=args.angles).to(args.device)
    x = torch.rand(args.batch_size, 1, args.image_size, args.image_size, device=args.device)
    y = projector(x).detach()

    def synchronize():
        if args.device == "mps":
            torch.mps.synchronize()
        elif args.device == "cuda":
            torch.cuda.synchronize()

    # An approximate norm is sufficient for a timing-only pass.
    norm = 2.18274 * (64 / args.image_size) ** 0.5 * (args.angles / 90) ** 0.5
    for name, model, inputs in (
        ("fbp-unet", FBPUNet(projector.circle_mask), x),
        ("lpd", LearnedPrimalDual(projector, norm), y),
    ):
        model.to(args.device)
        for step in range(args.steps):
            synchronize()
            start = time.perf_counter()
            model.zero_grad(set_to_none=True)
            output = model(inputs)
            output.square().mean().backward()
            synchronize()
            logging.info(
                "device=%s model=%s step=%d train_batch_seconds=%.4f",
                args.device,
                name,
                step,
                time.perf_counter() - start,
            )


if __name__ == "__main__":
    main()
