"""FBP filters for CT reconstruction."""

from typing import Literal

import torch
import torch.fft as fft

FilterType = Literal["ramp", "shepp-logan", "cosine", "hamming", "hann", "none"]
_FILTER_CACHE: dict[tuple[int, FilterType, bool, str, int | None, torch.dtype], torch.Tensor] = {}


def _filter_cache_key(
    size: int,
    filter_name: FilterType,
    device: torch.device | None,
    dtype: torch.dtype,
    real_fft: bool,
) -> tuple[int, FilterType, bool, str, int | None, torch.dtype]:
    if device is None:
        return (size, filter_name, real_fft, "cpu", None, dtype)
    return (size, filter_name, real_fft, device.type, device.index, dtype)


def _build_filter(
    size: int,
    filter_name: FilterType,
    device: torch.device | None,
    dtype: torch.dtype,
    *,
    real_fft: bool,
) -> torch.Tensor:
    if filter_name == "none":
        length = size // 2 + 1 if real_fft else size
        return torch.ones(length, device=device, dtype=dtype)

    freq_fn = fft.rfftfreq if real_fft else fft.fftfreq
    freq = freq_fn(size, d=1.0, device=device, dtype=torch.float32)
    ramp = freq.abs()

    if filter_name == "ramp":
        filt = ramp
    elif filter_name == "shepp-logan":
        filt = ramp * torch.sinc(2 * freq)
    elif filter_name == "cosine":
        filt = ramp * torch.cos(torch.pi * freq)
    elif filter_name == "hamming":
        filt = ramp * (0.54 + 0.46 * torch.cos(2 * torch.pi * freq))
    elif filter_name == "hann":
        filt = ramp * (0.5 + 0.5 * torch.cos(2 * torch.pi * freq))
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    return filt.to(dtype=dtype)


def _get_cached_filter(
    size: int,
    filter_name: FilterType,
    device: torch.device | None,
    dtype: torch.dtype,
    *,
    real_fft: bool,
) -> torch.Tensor:
    key = _filter_cache_key(size, filter_name, device, dtype, real_fft)
    filt = _FILTER_CACHE.get(key)
    if filt is None:
        filt = _build_filter(size, filter_name, device, dtype, real_fft=real_fft)
        _FILTER_CACHE[key] = filt
    return filt


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
    return _get_cached_filter(size, filter_name, device, dtype, real_fft=False)


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
    pad_len = max(64, 1 << (2 * n_det - 1).bit_length())

    filt = _get_cached_filter(pad_len, filter_name, device=device, dtype=dtype, real_fft=True)

    sino_fft = fft.rfft(sinogram, n=pad_len, dim=-1)

    filtered_fft = sino_fft * filt.view(1, 1, 1, -1)

    filtered = fft.irfft(filtered_fft, n=pad_len, dim=-1)

    filtered = filtered[..., :n_det]

    return filtered
