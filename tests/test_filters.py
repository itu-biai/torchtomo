"""Tests for FBP filters."""

import math

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

    def test_ramp_keeps_the_dc_bin(self):
        """The ramp is the DFT of the truncated Ram-Lak kernel, not a sampled `|f|`."""
        size = 1024
        filt = get_filter(size, "ramp")
        # Truncating Kak and Slaney's kernel to `size` taps leaves 2 / (pi^2 size)
        # in the DC bin, where sampling `|f|` leaves an exact zero and throws the
        # mean of every projection away.
        assert filt[0].item() == pytest.approx(2.0 / (math.pi**2 * size), rel=0.02)
        # Away from DC the two agree: this is the same ramp, built the other way.
        freq = torch.fft.fftfreq(size).abs()
        assert torch.allclose(filt[1:], freq[1:], rtol=0.03, atol=1e-4)

    def test_fbp_reprojects_to_its_own_measurements(self):
        """FBP inverts the forward projector: gain 1, no constant offset."""
        size, n_angles = 256, 180
        projector = ParallelBeam(img_size=size, n_angles=n_angles)
        phantom = shepp_logan(size)
        sinogram = projector.forward(phantom)
        recon = projector.fbp(sinogram)
        reprojected = projector.forward(recon)

        gain = ((reprojected * sinogram).sum() / sinogram.square().sum()).item()
        residual = ((reprojected - sinogram).norm() / sinogram.norm()).item()
        mask = projector.circle_mask.bool()
        bias = ((recon - phantom)[:, :, mask]).mean().item()

        assert gain == pytest.approx(1.0, abs=0.01)
        assert residual < 0.02
        assert abs(bias) < 1e-3

    def test_invalid_filter(self):
        with pytest.raises(ValueError):
            get_filter(256, "invalid_filter")
