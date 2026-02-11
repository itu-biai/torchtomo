"""
TorchTomo: Differentiable CT Reconstruction in Pure PyTorch

A lightweight library for CT forward and back projection that works
on any device (CPU, CUDA, MPS) without compilation.

Example:
    >>> from torchtomo import ParallelBeam, FanBeam
    >>>
    >>> # Parallel beam
    >>> projector = ParallelBeam(img_size=256, n_angles=180, n_det=256)
    >>> sinogram = projector.forward(image)
    >>> recon = projector.fbp(sinogram)
    >>>
    >>> # Fan beam
    >>> projector = FanBeam(img_size=256, n_angles=360, n_det=400,
    ...                     src_dist=500, det_dist=500)
    >>> sinogram = projector.forward(image)
    >>> recon = projector.fbp(sinogram)
"""

from .fanbeam import FanBeam
from .filters import apply_filter, get_filter
from .parallel import ParallelBeam
from .phantom import circle_phantom, shepp_logan

__version__ = "0.1.0"
__all__ = [
    "ParallelBeam",
    "FanBeam",
    "apply_filter",
    "get_filter",
    "shepp_logan",
    "circle_phantom",
]
