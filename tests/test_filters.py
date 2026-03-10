"""Tests for FBP filters."""

import pytest
import torch

from torchtomo import ParallelBeam, shepp_logan
from torchtomo.filters import apply_filter, get_filter


class TestFilters:
    @pytest.mark.parametrize("filter_name", ["ramp", "shepp-logan", "cosine", "hamming", "hann"])
    def test_filter_shape(self, filter_name):
        size = 256
        filt = get_filter(size, filter_name)
        assert filt.shape == (size,)

    def test_filter_none(self):
        size = 256
        filt = get_filter(size, "none")
        assert torch.allclose(filt, torch.ones(size))

    def test_apply_filter_shape(self):
        sinogram = torch.randn(2, 1, 90, 128)
        filtered = apply_filter(sinogram, "ramp")
        assert filtered.shape == sinogram.shape

    def test_apply_filter_none(self):
        sinogram = torch.randn(2, 1, 90, 128)
        filtered = apply_filter(sinogram, "none")
        assert torch.allclose(filtered, sinogram)

    def test_apply_filter_matches_manual_fft(self):
        sinogram = torch.randn(2, 1, 90, 128)
        pad_len = max(64, 1 << (2 * sinogram.shape[-1] - 1).bit_length())
        filt = get_filter(pad_len, "hamming", device=sinogram.device, dtype=sinogram.dtype)
        expected = torch.fft.ifft(
            torch.fft.fft(sinogram, n=pad_len, dim=-1) * filt.view(1, 1, 1, -1),
            dim=-1,
        ).real[..., : sinogram.shape[-1]]

        filtered = apply_filter(sinogram, "hamming")
        assert torch.allclose(filtered, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("filter_name", ["ramp", "shepp-logan", "cosine", "hamming", "hann"])
    def test_fbp_with_filters(self, filter_name):
        projector = ParallelBeam(img_size=128, n_angles=90)
        phantom = shepp_logan(128)
        sinogram = projector.forward(phantom)
        recon = projector.fbp(sinogram, filter_name=filter_name)
        assert recon.shape == phantom.shape

    def test_invalid_filter(self):
        with pytest.raises(ValueError):
            get_filter(256, "invalid_filter")
