"""FBP filters for CT reconstruction."""

from typing import Literal

import numpy as np
import torch
import torch.fft as fft

FilterType = Literal["ramp", "shepp-logan", "cosine", "hamming", "hann", "none"]


def get_filter(
    size: int,
    filter_name: FilterType = "ramp",
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate frequency-domain filter for FBP.

    Args:
        size: Filter size
        filter_name: Type of filter
        device: Target device
        dtype: Data type

    Returns:
        Filter in frequency domain, shape [size]
    """
    if filter_name == "none":
        return torch.ones(size, device=device, dtype=dtype)

    # Frequency axis: fftfreq gives [-0.5, 0.5) normalized frequencies
    freq = np.fft.fftfreq(size).astype(np.float64)

    # Ramp filter: |f| scaled to detector spacing
    # In FBP, the filter is |omega| = 2*pi*|f|, but with discrete sampling
    # we use |f| directly and scale appropriately
    ramp = np.abs(freq)

    if filter_name == "ramp":
        filt = ramp
    elif filter_name == "shepp-logan":
        # sinc window (avoids division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            window = np.sinc(2 * freq)  # sinc(2f) = sin(2*pi*f)/(2*pi*f)
        filt = ramp * window
    elif filter_name == "cosine":
        filt = ramp * np.cos(np.pi * freq)
    elif filter_name == "hamming":
        filt = ramp * (0.54 + 0.46 * np.cos(2 * np.pi * freq))
    elif filter_name == "hann":
        filt = ramp * (0.5 + 0.5 * np.cos(2 * np.pi * freq))
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    # Convert to torch tensor
    filt = torch.from_numpy(filt.astype(np.float32)).to(device=device, dtype=dtype)

    return filt


def apply_filter(
    sinogram: torch.Tensor,
    filter_name: FilterType = "ramp",
) -> torch.Tensor:
    """
    Apply FBP filter to sinogram in frequency domain.

    Args:
        sinogram: Sinogram of shape [B, 1, n_angles, n_det]
        filter_name: Type of filter to apply

    Returns:
        Filtered sinogram of shape [B, 1, n_angles, n_det]
    """
    if filter_name == "none":
        return sinogram

    B, C, n_angles, n_det = sinogram.shape
    device = sinogram.device
    dtype = sinogram.dtype

    # Pad to next power of 2 for efficient FFT (and to avoid circular conv)
    pad_len = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))

    # Get filter
    filt = get_filter(pad_len, filter_name, device=device, dtype=dtype)

    # FFT of sinogram with zero-padding
    sino_fft = fft.fft(sinogram, n=pad_len, dim=-1)

    # Apply filter in frequency domain
    filtered_fft = sino_fft * filt.view(1, 1, 1, -1)

    # Inverse FFT
    filtered = fft.ifft(filtered_fft, dim=-1).real

    # Crop to original size
    filtered = filtered[..., :n_det]

    return filtered
