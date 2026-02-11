"""Tests for batch processing."""

import torch

from torchtomo import circle_phantom, shepp_logan


class TestBatchProcessing:
    def test_parallel_batch(self, parallel_projector):
        batch = torch.cat(
            [
                shepp_logan(128),
                circle_phantom(128),
            ],
            dim=0,
        )
        assert batch.shape == (2, 1, 128, 128)
        sinogram = parallel_projector.forward(batch)
        assert sinogram.shape[0] == 2
        recon = parallel_projector.fbp(sinogram)
        assert recon.shape[0] == 2

    def test_fan_batch(self, fan_projector):
        batch = torch.cat(
            [
                shepp_logan(128),
                circle_phantom(128),
            ],
            dim=0,
        )
        assert batch.shape == (2, 1, 128, 128)
        sinogram = fan_projector.forward(batch)
        assert sinogram.shape[0] == 2
        recon = fan_projector.fbp(sinogram)
        assert recon.shape[0] == 2

    def test_parallel_larger_batch(self, parallel_projector):
        batch_size = 4
        batch = shepp_logan(128).expand(batch_size, -1, -1, -1)
        sinogram = parallel_projector.forward(batch)
        assert sinogram.shape[0] == batch_size
        recon = parallel_projector.fbp(sinogram)
        assert recon.shape[0] == batch_size
