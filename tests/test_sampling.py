"""Check the MPS sampling formula against PyTorch's native CPU operator."""

import pytest
import torch
import torch.nn.functional as F

from torchtomo._sampling import _gather_bilinear


@pytest.mark.parametrize("shape", [(5, 7), (1, 7), (7, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gather_sampling_matches_native_values_and_image_gradients(shape, dtype):
    torch.manual_seed(10)
    image = torch.randn(2, 3, *shape, dtype=dtype, requires_grad=True)
    grid = torch.rand(2, 4, 6, 2, dtype=dtype) * 3 - 1.5
    grid[:, 0, :3] = torch.tensor([[-1, -1], [1, 1], [0, 0]], dtype=dtype)
    upstream = torch.randn(2, 3, 4, 6, dtype=dtype)
    actual = _gather_bilinear(image, grid)
    expected = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, image, upstream)[0]
    expected_grad = torch.autograd.grad(expected, image, upstream)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
