"""Test phantoms for CT reconstruction."""

from typing import Optional

import numpy as np
import torch


def circle_phantom(
    size: int = 256,
    n_circles: int = 5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a simple phantom with random circles.

    Args:
        size: Image size
        n_circles: Number of circles
        device: Target device

    Returns:
        Phantom image [1, 1, size, size]
    """
    coords = torch.linspace(-1, 1, size, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")

    phantom = torch.zeros(size, size, device=device)

    # Background circle
    phantom += 0.2 * ((x**2 + y**2) < 0.9).float()

    # Random circles
    torch.manual_seed(42)
    for _ in range(n_circles):
        cx = torch.rand(1).item() * 1.2 - 0.6
        cy = torch.rand(1).item() * 1.2 - 0.6
        r = torch.rand(1).item() * 0.2 + 0.05
        intensity = torch.rand(1).item() * 0.8 + 0.2

        circle = ((x - cx) ** 2 + (y - cy) ** 2) < r**2
        phantom += intensity * circle.float()

    # Clip to [0, 1]
    phantom = phantom.clamp(0, 1)

    return phantom.unsqueeze(0).unsqueeze(0)


def shepp_logan(
    size: int = 256,
    device: Optional[torch.device] = None,
    modified: bool = True,
) -> torch.Tensor:
    """
    Create the Shepp-Logan phantom.

    The Shepp-Logan phantom is a standard test image for CT reconstruction
    algorithms. It consists of ellipses simulating a human head.

    Args:
        size: Image size
        device: Target device
        modified: If True, use modified (higher contrast) version

    Returns:
        Phantom image [1, 1, size, size]
    """
    # Ellipse parameters: (intensity, a, b, x0, y0, phi)
    # a, b = semi-axes, x0, y0 = center, phi = rotation angle

    if modified:
        # Modified Shepp-Logan with better contrast
        ellipses = [
            (1.0, 0.69, 0.92, 0, 0, 0),  # Outer skull
            (-0.8, 0.6624, 0.874, 0, -0.0184, 0),  # Brain
            (-0.2, 0.11, 0.31, 0.22, 0, -18),  # Left ventricle
            (-0.2, 0.16, 0.41, -0.22, 0, 18),  # Right ventricle
            (0.1, 0.21, 0.25, 0, 0.35, 0),  # Top feature
            (0.1, 0.046, 0.046, 0, 0.1, 0),  # Small circle 1
            (0.1, 0.046, 0.046, 0, -0.1, 0),  # Small circle 2
            (0.1, 0.046, 0.023, -0.08, -0.605, 0),  # Bottom left
            (0.1, 0.023, 0.023, 0, -0.606, 0),  # Bottom center
            (0.1, 0.023, 0.046, 0.06, -0.605, 0),  # Bottom right
        ]
    else:
        # Original Shepp-Logan (low contrast)
        ellipses = [
            (2.0, 0.69, 0.92, 0, 0, 0),
            (-0.98, 0.6624, 0.874, 0, -0.0184, 0),
            (-0.02, 0.11, 0.31, 0.22, 0, -18),
            (-0.02, 0.16, 0.41, -0.22, 0, 18),
            (0.01, 0.21, 0.25, 0, 0.35, 0),
            (0.01, 0.046, 0.046, 0, 0.1, 0),
            (0.01, 0.046, 0.046, 0, -0.1, 0),
            (0.01, 0.046, 0.023, -0.08, -0.605, 0),
            (0.01, 0.023, 0.023, 0, -0.606, 0),
            (0.01, 0.023, 0.046, 0.06, -0.605, 0),
        ]

    # Create coordinate grid
    coords = torch.linspace(-1, 1, size, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    y = -y  # Flip y to match standard orientation (details at bottom)

    phantom = torch.zeros(size, size, device=device)

    for intensity, a, b, x0, y0, phi in ellipses:
        phi_rad = phi * np.pi / 180

        # Rotate coordinates
        cos_p = np.cos(phi_rad)
        sin_p = np.sin(phi_rad)

        x_rot = cos_p * (x - x0) + sin_p * (y - y0)
        y_rot = -sin_p * (x - x0) + cos_p * (y - y0)

        # Ellipse equation
        inside = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1

        phantom += intensity * inside.float()

    # Normalize to [0, 1]
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-8)

    return phantom.unsqueeze(0).unsqueeze(0)


def forbild(
    size: int = 256,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a simplified FORBILD head phantom.

    A more challenging phantom with fine details for testing
    resolution and artifact performance.

    Args:
        size: Image size
        device: Target device

    Returns:
        Phantom image [1, 1, size, size]
    """
    coords = torch.linspace(-1, 1, size, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")

    phantom = torch.zeros(size, size, device=device)

    # Outer skull (ellipse)
    skull = ((x / 0.85) ** 2 + (y / 0.95) ** 2) < 1
    phantom += 0.2 * skull.float()

    # Brain tissue
    brain = ((x / 0.75) ** 2 + (y / 0.85) ** 2) < 1
    phantom += 0.3 * brain.float()

    # Ventricles (pair of ellipses)
    vent_l = (((x - 0.2) / 0.08) ** 2 + ((y - 0.1) / 0.25) ** 2) < 1
    vent_r = (((x + 0.2) / 0.08) ** 2 + ((y - 0.1) / 0.25) ** 2) < 1
    phantom -= 0.3 * (vent_l | vent_r).float()

    # High-contrast inserts (simulating lesions)
    for i in range(5):
        cx = 0.4 * np.cos(2 * np.pi * i / 5)
        cy = 0.4 * np.sin(2 * np.pi * i / 5) - 0.1
        r = 0.05

        insert = ((x - cx) ** 2 + (y - cy) ** 2) < r**2
        phantom += 0.5 * insert.float()

    # Fine resolution pattern (line pairs)
    for i, offset in enumerate([0.6, 0.65, 0.7, 0.75]):
        width = 0.02 / (i + 1)
        for j in range(3):
            cx = offset
            cy = -0.3 + j * width * 3

            line = (torch.abs(x - cx) < width) & (torch.abs(y - cy) < width)
            phantom += 0.4 * line.float()

    # Normalize
    phantom = phantom.clamp(0, 1)

    return phantom.unsqueeze(0).unsqueeze(0)
