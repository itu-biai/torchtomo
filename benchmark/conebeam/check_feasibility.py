"""Synthetic 3D helical feasibility checks, independent of the public 2D API."""

import argparse
import itertools
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Geometry:
    size: int = 32
    views: int = 48
    rows: int = 16
    columns: int = 48
    samples: int = 96
    voxel_spacing: float = 1.0
    detector_spacing: float = 2.0
    source_radius: float = 80.0
    source_detector_distance: float = 160.0
    axial_travel: float = 8.0
    turns: float = 1.5
    ray_chunk: int = 512


class HelicalReference:
    def __init__(
        self,
        geometry: Geometry,
        device: str,
        dtype: torch.dtype,
        detector_shift: torch.Tensor | None = None,
    ) -> None:
        self.geometry = geometry
        self.device = device
        self.dtype = dtype
        self.detector_shift = torch.zeros((), device=device, dtype=dtype) if detector_shift is None else detector_shift
        self.corners = torch.tensor(list(itertools.product((0, 1), repeat=3)), device=device)
        self.fractions = (torch.arange(geometry.samples, device=device, dtype=dtype) + 0.5) / geometry.samples

    def rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = self.geometry
        angles = torch.arange(geometry.views, device=self.device, dtype=self.dtype)
        angles = angles * (2 * math.pi * geometry.turns / geometry.views)
        axial = torch.linspace(-0.5, 0.5, geometry.views, device=self.device, dtype=self.dtype)
        axial = axial * geometry.axial_travel
        zeros = torch.zeros_like(angles)
        source = torch.stack(
            (geometry.source_radius * angles.cos(), geometry.source_radius * angles.sin(), axial), dim=-1
        )
        inward = torch.stack((-angles.cos(), -angles.sin(), zeros), dim=-1)
        tangent = torch.stack((-angles.sin(), angles.cos(), zeros), dim=-1)
        vertical = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.dtype)
        column = torch.arange(geometry.columns, device=self.device, dtype=self.dtype)
        fan_angle = (column - (geometry.columns - 1) / 2) * (
            geometry.detector_spacing / geometry.source_detector_distance
        )
        row = torch.arange(geometry.rows, device=self.device, dtype=self.dtype)
        height = (row - (geometry.rows - 1) / 2) * geometry.detector_spacing
        radial = fan_angle.cos()[None, :, None] * inward[:, None, :]
        radial = radial + fan_angle.sin()[None, :, None] * tangent[:, None, :]
        detector = source[:, None, None, :] + geometry.source_detector_distance * radial[:, None, :, :]
        detector = detector + height[None, :, None, None] * vertical
        detector = detector + self.detector_shift * tangent[:, None, None, :]
        origins = source[:, None, None, :].expand_as(detector)
        directions = detector - origins
        directions = directions / directions.norm(dim=-1, keepdim=True)
        return origins.reshape(-1, 3), directions.reshape(-1, 3)

    def stencils(self, origins: torch.Tensor, directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = self.geometry
        half_extent = geometry.size * geometry.voxel_spacing / 2
        moving = directions.abs() > torch.finfo(self.dtype).eps
        safe_directions = torch.where(moving, directions, torch.ones_like(directions))
        near = (-half_extent - origins) / safe_directions
        far = (half_extent - origins) / safe_directions
        inside = origins.abs() <= half_extent
        lower = torch.where(moving, torch.minimum(near, far), -torch.inf)
        upper = torch.where(moving, torch.maximum(near, far), torch.where(inside, torch.inf, -torch.inf))
        entry = lower.amax(dim=-1).clamp_min(0)
        length = (upper.amin(dim=-1) - entry).clamp_min(0)
        distance = entry[:, None] + length[:, None] * self.fractions
        points = origins[:, None, :] + distance[..., None] * directions[:, None, :]
        coordinates = points / geometry.voxel_spacing + (geometry.size - 1) / 2
        base = coordinates.floor().long()
        fraction = coordinates - base.to(self.dtype)
        indices = base[..., None, :] + self.corners
        valid = ((indices >= 0) & (indices < geometry.size)).all(dim=-1)
        weights = torch.where(self.corners.bool(), fraction[..., None, :], 1 - fraction[..., None, :])
        weights = weights.prod(dim=-1) * valid * (length[:, None, None] / geometry.samples)
        indices = indices.clamp(0, geometry.size - 1)
        flat_indices = indices[..., 0] + geometry.size * (indices[..., 1] + geometry.size * indices[..., 2])
        return flat_indices, weights

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry
        origins, directions = self.rays()
        flat_volume = volume.reshape(volume.shape[0], -1)
        chunks = []
        for start in range(0, origins.shape[0], geometry.ray_chunk):
            end = start + geometry.ray_chunk
            indices, weights = self.stencils(origins[start:end], directions[start:end])
            values = flat_volume[:, indices.flatten()].reshape(volume.shape[0], *indices.shape)
            chunks.append((values * weights).sum(dim=(-1, -2)))
        return torch.cat(chunks, dim=-1).reshape(volume.shape[0], 1, geometry.views, geometry.rows, geometry.columns)

    def adjoint(self, projections: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry
        origins, directions = self.rays()
        batch = projections.shape[0]
        flat_projections = projections.reshape(batch, -1)
        image = projections.new_zeros(batch, geometry.size**3)
        for start in range(0, origins.shape[0], geometry.ray_chunk):
            end = start + geometry.ray_chunk
            indices, weights = self.stencils(origins[start:end], directions[start:end])
            values = flat_projections[:, start:end, None, None] * weights
            image = image.scatter_add(1, indices.flatten()[None, :].expand(batch, -1), values.reshape(batch, -1))
        return image.reshape(batch, 1, geometry.size, geometry.size, geometry.size)


def gaussian_volume(projector: HelicalReference, sigma: float) -> torch.Tensor:
    geometry = projector.geometry
    axis = torch.arange(geometry.size, device=projector.device, dtype=projector.dtype)
    axis = (axis - (geometry.size - 1) / 2) * geometry.voxel_spacing
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.exp(-(x.square() + y.square() + z.square()) / (2 * sigma**2))[None, None]


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.detach() - expected.detach()
    return float(difference.norm() / expected.detach().norm().clamp_min(torch.finfo(expected.dtype).eps))


def check_adjoint_and_gradients(device: str) -> dict:
    geometry = Geometry(size=5, views=7, rows=4, columns=6, samples=23, ray_chunk=64)
    projector = HelicalReference(geometry, device, torch.float64)
    basis = torch.eye(geometry.size**3, device=device, dtype=torch.float64)
    with torch.no_grad():
        matrix = projector.forward(basis.reshape(-1, 1, geometry.size, geometry.size, geometry.size)).flatten(1).T
    x = torch.randn(2, 1, geometry.size, geometry.size, geometry.size, device=device, dtype=torch.float64)
    y = torch.randn(2, 1, geometry.views, geometry.rows, geometry.columns, device=device, dtype=torch.float64)
    x.requires_grad_()
    y.requires_grad_()
    ax, aty = projector.forward(x), projector.adjoint(y)
    expected = (y.flatten(1) @ matrix).reshape_as(x)
    matrix_error = relative_error(aty, expected)
    scale = ax.norm() * y.norm() + x.norm() * aty.norm()
    dot_error = float(((ax * y).sum() - (x * aty).sum()).abs().detach() / scale.detach())
    forward_vjp = torch.autograd.grad(ax, x, y)[0]
    adjoint_vjp = torch.autograd.grad(aty, y, x)[0]
    forward_gradient_error = relative_error(forward_vjp, aty.detach())
    adjoint_gradient_error = relative_error(adjoint_vjp, ax.detach())
    shift = torch.tensor(0.23, device=device, dtype=torch.float64, requires_grad=True)

    def objective(value: torch.Tensor, operator: str) -> torch.Tensor:
        shifted = HelicalReference(geometry, device, torch.float64, value)
        if operator == "forward":
            return (shifted.forward(x.detach()) * y.detach()).sum()
        return (shifted.adjoint(y.detach()) * x.detach()).sum()

    geometry_errors = {}
    for operator in ("forward", "adjoint"):
        derivative = torch.autograd.grad(objective(shift, operator), shift)[0]
        step = 1e-5
        with torch.no_grad():
            finite_difference = (objective(shift + step, operator) - objective(shift - step, operator)) / (2 * step)
        geometry_errors[operator] = float(
            (derivative - finite_difference).abs() / finite_difference.abs().clamp_min(1e-8)
        )
    results = {
        "matrix_transpose_relative_error": matrix_error,
        "normalized_dot_error": dot_error,
        "forward_vjp_relative_error": forward_gradient_error,
        "adjoint_vjp_relative_error": adjoint_gradient_error,
        "geometry_gradient_relative_errors": geometry_errors,
    }
    if max(matrix_error, dot_error, forward_gradient_error, adjoint_gradient_error) > 1e-10:
        raise RuntimeError(f"Adjoint or input gradient check failed: {results}")
    if max(geometry_errors.values()) > 1e-4:
        raise RuntimeError(f"Geometry gradient check failed: {results}")
    return results


@torch.no_grad()
def check_reconstruction(device: str, output: Path) -> dict:
    geometry = Geometry()
    projector = HelicalReference(geometry, device, torch.float32)
    sigma = 3.0
    volume = gaussian_volume(projector, sigma)
    origins, directions = projector.rays()
    distance_squared = torch.cross(origins, directions, dim=-1).square().sum(dim=-1)
    analytic = math.sqrt(2 * math.pi) * sigma * torch.exp(-distance_squared / (2 * sigma**2))
    projections = projector.forward(volume)
    analytic_error = relative_error(projections.flatten(), analytic)
    reconstruction = torch.zeros_like(volume)
    residual = projections.clone()
    gradient = projector.adjoint(residual)
    direction = gradient.clone()
    gamma = gradient.square().sum()
    history = [1.0]
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(15):
        projected = projector.forward(direction)
        alpha = gamma / projected.square().sum().clamp_min(1e-20)
        reconstruction += alpha * direction
        residual -= alpha * projected
        next_gradient = projector.adjoint(residual)
        next_gamma = next_gradient.square().sum()
        direction = next_gradient + (next_gamma / gamma.clamp_min(1e-20)) * direction
        gamma = next_gamma
        history.append(float(residual.norm() / projections.norm()))
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    image_error = relative_error(reconstruction, volume)
    torch.save(
        {"reference": volume.cpu(), "reconstruction": reconstruction.cpu(), "projections": projections.cpu()},
        output / "synthetic_reconstruction.pt",
    )
    results = {
        "geometry": asdict(geometry),
        "analytic_gaussian_projection_relative_error": analytic_error,
        "cgls_iterations": 15,
        "cgls_seconds": seconds,
        "projection_residual_history": history,
        "volume_relative_error": image_error,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
    }
    if analytic_error > 0.03 or history[-1] > 0.05 or image_error > 0.25:
        raise RuntimeError(f"Synthetic accuracy check failed: {results}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=Path("/content/torchtomo_conebeam_results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    if args.device == "cpu":
        torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this run")
    results = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "scope": "Synthetic feasibility only; TCIA files and native CUDA kernels are not evaluated",
        "correctness": check_adjoint_and_gradients(args.device),
        "reconstruction": check_reconstruction(args.device, args.output),
    }
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
