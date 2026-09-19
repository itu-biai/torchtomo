"""Compare torchtomo, LEAP, and torch-radon on the same parallel- or fan-beam problem.

Four dimensions: reconstruction quality (PSNR and SSIM against the phantom),
agreement between the libraries' sinograms, speed, and GPU memory footprint.
A figure of the reconstructions and their error maps is written alongside.

Each library projects the phantom and reconstructs it with its own conventions,
so the scale each one works in cancels and the quality figures are comparable
without any fitted correction. All three are given the same angle list, the same
phantom, and the same inscribed circle to be scored over.

    PYTHONPATH=src:benchmark python benchmark/compare_libraries.py --output /tmp/cmp

Speed and memory are only meaningful on an idle card; --sections lets the quality
work run while something else is using the GPU.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from torchtomo import FanBeam, ParallelBeam, shepp_logan
from torchtomo._nvrtc import runtime_available
from torchtomo.phantom import circle_phantom

# torch-radon 1.0 predates two removals it depends on: np.int, and torch.rfft.
# Both are restored here rather than edited into site-packages, so the installed
# package stays as it was shipped and its own filter definition is still used.
if not hasattr(np, "int"):
    np.int = int

try:
    from torch_radon import Radon, RadonFanbeam
except ImportError:
    Radon = None
    RadonFanbeam = None

try:
    from leap_projector import LeapFanBeam, LeapParallelBeam
except ImportError:
    LeapParallelBeam = None
    LeapFanBeam = None

try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None


def psnr(truth, image, mask, data_range=1.0):
    """Mean squared error over the visible circle only, as decibels."""
    error = ((image - truth) ** 2 * mask).sum() / mask.sum()
    return float(10.0 * torch.log10(torch.as_tensor(data_range**2) / error))


def ssim(truth, image, mask, data_range=1.0):
    reference = (truth * mask).squeeze().cpu().numpy()
    other = (image * mask).squeeze().cpu().numpy()
    return float(structural_similarity(reference, other, data_range=data_range))


class TorchtomoBackend:
    name = "torchtomo"

    def __init__(self, size, angles, device):
        self.projector = ParallelBeam(img_size=size, n_angles=len(angles), angles=angles).to(device)

    @property
    def angles(self):
        return self.projector.angles

    def forward(self, image):
        return self.projector.forward(image)

    def backproject(self, sinogram):
        return self.projector.backward(sinogram)

    def fbp(self, sinogram):
        return self.projector.fbp(sinogram)


class TorchtomoCudaBackend(TorchtomoBackend):
    """torchtomo with backend="cuda": the runtime-compiled kernels."""

    name = "torchtomo-cuda"

    def __init__(self, size, angles, device):
        self.projector = ParallelBeam(img_size=size, n_angles=len(angles), angles=angles, backend="cuda").to(device)


class LeapBackend:
    name = "leap"

    def __init__(self, size, angles, device):
        self.projector = LeapParallelBeam(img_size=size, n_angles=len(angles), angles=angles).to(device)

    def forward(self, image):
        return self.projector.forward(image)

    def backproject(self, sinogram):
        return self.projector.backward(sinogram)

    def fbp(self, sinogram):
        return self.projector.fbp(sinogram)


class TorchRadonBackend:
    name = "torch-radon"

    def __init__(self, size, angles, device):
        self.radon = Radon(size, angles.detach().cpu().numpy(), clip_to_circle=True)
        self.size = size

    def forward(self, image):
        return self.radon.forward(image)

    def backproject(self, sinogram):
        return self.radon.backprojection(sinogram)

    def filter_sinogram(self, sinogram, filter_name="ramp"):
        """torch-radon's own filter, on the FFT API that replaced torch.rfft.

        The filter itself still comes from torch-radon, so only the transform
        calls differ from what the library shipped.
        """
        batch, channels, n_angles, detectors = sinogram.shape
        flat = sinogram.reshape(batch * channels, n_angles, detectors).float()
        padded_size = max(64, int(2 ** np.ceil(np.log2(2 * detectors))))
        pad = padded_size - detectors
        padded = F.pad(flat, (0, pad, 0, 0))
        spectrum = torch.fft.fft(padded, dim=-1, norm="ortho")
        kernel = self.radon.fourier_filters.get(padded_size, filter_name, sinogram.device).view(1, 1, -1)
        filtered = torch.fft.ifft(spectrum * kernel, dim=-1, norm="ortho").real
        filtered = filtered[:, :, :-pad] * (np.pi / (2 * n_angles))
        return filtered.reshape(batch, channels, n_angles, detectors)

    def fbp(self, sinogram):
        return self.radon.backprojection(self.filter_sinogram(sinogram))


class TorchtomoFanBackend:
    name = "torchtomo"

    def __init__(self, size, angles, device):
        self.projector = FanBeam(img_size=size, n_angles=len(angles), angles=angles).to(device)

    def forward(self, image):
        return self.projector.forward(image)

    def backproject(self, sinogram):
        return self.projector.backward(sinogram)

    def fbp(self, sinogram):
        return self.projector.fbp(sinogram)


class TorchtomoCudaFanBackend(TorchtomoFanBackend):
    """torchtomo FanBeam with backend="cuda": the runtime-compiled kernels."""

    name = "torchtomo-cuda"

    def __init__(self, size, angles, device):
        self.projector = FanBeam(img_size=size, n_angles=len(angles), angles=angles, backend="cuda").to(device)


class LeapFanBackend:
    name = "leap"

    def __init__(self, size, angles, device):
        self.projector = LeapFanBeam(img_size=size, n_angles=len(angles), angles=angles).to(device)

    def forward(self, image):
        return self.projector.forward(image)

    def backproject(self, sinogram):
        return self.projector.backward(sinogram)

    def fbp(self, sinogram):
        return self.projector.fbp(sinogram)


class TorchRadonFanBackend:
    name = "torch-radon"

    def __init__(self, size, angles, device):
        reference = FanBeam(img_size=size, n_angles=len(angles), angles=angles.cpu())
        spacing = reference.det_width / reference.n_det
        self.radon = RadonFanbeam(
            size,
            angles.detach().cpu().numpy(),
            source_distance=reference.src_dist,
            det_distance=reference.det_dist,
            det_count=reference.n_det,
            det_spacing=spacing,
            clip_to_circle=True,
        )
        self.size = size

    def forward(self, image):
        return self.radon.forward(image)

    def backproject(self, sinogram):
        return self.radon.backprojection(sinogram)

    def filter_sinogram(self, sinogram, filter_name="ramp"):
        return TorchRadonBackend.filter_sinogram(self, sinogram, filter_name)

    def fbp(self, sinogram):
        return self.radon.backprojection(self.filter_sinogram(sinogram))


def available_backends(geometry="parallel"):
    cuda_kernels = torch.cuda.is_available() and runtime_available()
    if geometry == "fan":
        backends = [TorchtomoFanBackend]
        if cuda_kernels:
            backends.append(TorchtomoCudaFanBackend)
        if LeapFanBeam is not None:
            backends.append(LeapFanBackend)
        if RadonFanbeam is not None:
            backends.append(TorchRadonFanBackend)
        return backends
    backends = [TorchtomoBackend]
    if cuda_kernels:
        backends.append(TorchtomoCudaBackend)
    if LeapParallelBeam is not None:
        backends.append(LeapBackend)
    if Radon is not None:
        backends.append(TorchRadonBackend)
    return backends


def angle_tensor(n_angles, device, geometry):
    span = 2 * np.pi if geometry == "fan" else np.pi
    return torch.arange(n_angles, device=device, dtype=torch.float32) * (span / n_angles)


def make_phantom(name, size, device):
    if name == "shepp-logan":
        return shepp_logan(size).reshape(1, 1, size, size).to(device)
    if name == "disc":
        coords = torch.linspace(-1, 1, size, device=device)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        return ((grid_x**2 + grid_y**2) < 0.3**2).float().reshape(1, 1, size, size)
    if name == "circles":
        return circle_phantom(size=size, device=device).reshape(1, 1, size, size)
    raise ValueError(f"unknown phantom {name}")


def measure_quality(args, device, results):
    """Each library projects the phantom and reconstructs its own sinogram."""
    for size in args.sizes:
        reference = ParallelBeam(img_size=size, n_angles=args.angles[0]).to(device)
        mask = reference.circle_mask.view(1, 1, size, size)
        for n_angles in args.angles:
            angles = angle_tensor(n_angles, device, args.geometry)
            for phantom_name in args.phantoms:
                truth = make_phantom(phantom_name, size, device) * mask
                sinograms = {}
                for backend_class in available_backends(args.geometry):
                    backend = backend_class(size, angles, device)
                    sinogram = backend.forward(truth)
                    reconstruction = backend.fbp(sinogram) * mask
                    sinograms[backend.name] = sinogram
                    row = {
                        "library": backend.name,
                        "size": size,
                        "angles": n_angles,
                        "phantom": phantom_name,
                        "psnr_db": psnr(truth, reconstruction, mask),
                        "ssim": ssim(truth, reconstruction, mask) if structural_similarity else None,
                    }
                    results["quality"].append(row)
                    print(
                        f"  quality {backend.name:12s} {size}px {n_angles:3d} angles {phantom_name:12s} "
                        f"psnr={row['psnr_db']:6.2f} dB ssim={row['ssim']:.4f}"
                    )
                base = sinograms["torchtomo"]
                for name, sinogram in sinograms.items():
                    if name == "torchtomo":
                        continue
                    scale = float((base * sinogram).sum() / (sinogram * sinogram).sum())
                    relative = float((base - sinogram * scale).norm() / base.norm())
                    results["agreement"].append(
                        {
                            "library": name,
                            "size": size,
                            "angles": n_angles,
                            "phantom": phantom_name,
                            "scale_to_torchtomo": scale,
                            "relative_l2_after_scale": relative,
                        }
                    )
                    print(f"  sinogram {name:12s} vs torchtomo: scale={scale:.6f} relative_l2={relative:.6f}")


def timed(function, warmup=3, runs=20):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs


def device_used_bytes():
    free, total = torch.cuda.mem_get_info()
    return total - free


def measure_speed_and_memory(args, device, results):
    """Time each operator and record what it costs the card.

    LEAP allocates outside PyTorch's caching allocator, so torch's own counters
    cannot see it. The driver figure is the one that covers all three libraries.
    """
    for size in args.sizes:
        for n_angles in args.angles:
            angles = angle_tensor(n_angles, device, args.geometry)
            for batch in args.batches:
                truth = make_phantom("shepp-logan", size, device).repeat(batch, 1, 1, 1)
                for backend_class in available_backends(args.geometry):
                    backend = backend_class(size, angles, device)
                    sinogram = backend.forward(truth).detach()
                    operations = {
                        "forward": lambda: backend.forward(truth),
                        "backproject": lambda: backend.backproject(sinogram),
                        "fbp": lambda: backend.fbp(sinogram),
                    }
                    for operation, function in operations.items():
                        function()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        before_driver = device_used_bytes()
                        before_torch = torch.cuda.memory_allocated()
                        function()
                        torch.cuda.synchronize()
                        driver_delta = device_used_bytes() - before_driver
                        torch_peak = torch.cuda.max_memory_allocated() - before_torch
                        seconds = timed(function, warmup=args.warmup, runs=args.runs)
                        row = {
                            "library": backend.name,
                            "operation": operation,
                            "size": size,
                            "angles": n_angles,
                            "batch": batch,
                            "milliseconds": seconds * 1e3,
                            "images_per_second": batch / seconds,
                            "torch_peak_mb": torch_peak / 1e6,
                            "driver_delta_mb": driver_delta / 1e6,
                        }
                        results["performance"].append(row)
                        print(
                            f"  speed {backend.name:12s} {operation:12s} {size}px {n_angles:3d}a "
                            f"batch={batch} {row['milliseconds']:8.3f} ms "
                            f"{row['images_per_second']:8.1f} img/s "
                            f"torch_peak={row['torch_peak_mb']:7.1f} MB driver={row['driver_delta_mb']:7.1f} MB"
                        )
                    torch.cuda.empty_cache()


def save_figure(args, device, path):
    """Reconstructions and error maps, one row per library."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size, n_angles = args.figure_size, args.figure_angles
    angles = angle_tensor(n_angles, device, args.geometry)
    reference = (
        FanBeam(img_size=size, n_angles=n_angles)
        if args.geometry == "fan"
        else ParallelBeam(img_size=size, n_angles=n_angles)
    ).to(device)
    mask = reference.circle_mask.view(1, 1, size, size)
    truth = make_phantom("shepp-logan", size, device) * mask

    backends = available_backends(args.geometry)
    figure, axes = plt.subplots(len(backends), 4, figsize=(13, 3.3 * len(backends)))
    axes = np.atleast_2d(axes)
    for row, backend_class in enumerate(backends):
        backend = backend_class(size, angles, device)
        sinogram = backend.forward(truth)
        reconstruction = backend.fbp(sinogram) * mask
        quality = psnr(truth, reconstruction, mask)
        structure = ssim(truth, reconstruction, mask) if structural_similarity else float("nan")
        image = reconstruction.squeeze().detach().cpu().numpy()
        error = (reconstruction - truth).squeeze().detach().cpu().numpy()
        strip = sinogram.squeeze().detach().cpu().numpy()

        axes[row, 0].imshow(truth.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(backend.name, fontsize=12)
        axes[row, 0].set_title("phantom" if row == 0 else "")
        axes[row, 1].imshow(strip, cmap="gray", aspect="auto")
        axes[row, 1].set_title("sinogram" if row == 0 else "")
        axes[row, 2].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("reconstruction" if row == 0 else "")
        axes[row, 2].set_xlabel(f"{quality:.2f} dB, SSIM {structure:.4f}")
        shown = axes[row, 3].imshow(error, cmap="coolwarm", vmin=-0.2, vmax=0.2)
        axes[row, 3].set_title("error against phantom" if row == 0 else "")
        figure.colorbar(shown, ax=axes[row, 3], fraction=0.046)
        for column in range(4):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

    figure.suptitle(f"{args.geometry.capitalize()} beam, {size} px, {n_angles} angles, ramp filter", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    print(f"  figure written to {path}")


COLORS = {"torchtomo": "#4C72B0", "leap": "#DD8452", "torch-radon": "#55A868"}


def save_summary_figure(results, path):
    """Quality, speed, and memory side by side, from what the run recorded."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    libraries = results["libraries"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    def grouped(axis, rows, key, label, configs, formatter):
        width = 0.8 / len(libraries)
        positions = np.arange(len(configs))
        for index, library in enumerate(libraries):
            values = []
            for config in configs:
                match = [r for r in rows if r["library"] == library and formatter(r) == config]
                values.append(match[0][key] if match else 0.0)
            axis.bar(positions + index * width, values, width, label=library, color=COLORS.get(library))
        axis.set_xticks(positions + width * (len(libraries) - 1) / 2)
        axis.set_xticklabels(configs, rotation=20, ha="right", fontsize=8)
        axis.set_ylabel(label)
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.3)

    quality = [r for r in results["quality"] if r["phantom"] == "shepp-logan"]

    def shape(row):
        return f"{row['size']}px/{row['angles']}a"

    configs = sorted({shape(r) for r in quality}, key=lambda c: (int(c.split("px")[0]), int(c.split("/")[1][:-1])))
    grouped(axes[0, 0], quality, "psnr_db", "PSNR (dB)", configs, shape)
    axes[0, 0].set_title("Reconstruction quality, Shepp-Logan")
    grouped(axes[0, 1], quality, "ssim", "SSIM", configs, shape)
    axes[0, 1].set_title("Structural similarity, Shepp-Logan")

    performance = [r for r in results["performance"] if r["size"] == 512 and r["angles"] == 360 and r["batch"] == 4]
    operations = ["forward", "backproject", "fbp"]
    grouped(axes[1, 0], performance, "images_per_second", "images per second", operations, lambda r: r["operation"])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Throughput, 512 px, 360 angles, batch 4")
    grouped(axes[1, 1], performance, "torch_peak_mb", "peak MB", operations, lambda r: r["operation"])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("GPU memory for one call, 512 px, 360 angles, batch 4")

    figure.suptitle(f"torchtomo against LEAP and torch-radon, {results['device']}", fontsize=13)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    print(f"  summary figure written to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--angles", type=int, nargs="+", default=[90, 180, 360])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--phantoms", nargs="+", default=["shepp-logan", "disc"])
    parser.add_argument("--sections", nargs="+", default=["quality", "performance", "figure"])
    parser.add_argument("--figure-size", type=int, default=512)
    parser.add_argument("--figure-angles", type=int, default=90)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--geometry", choices=("parallel", "fan"), default="parallel")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this comparison needs CUDA: torch-radon and LEAP are CUDA only")
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    names = [backend.name for backend in available_backends(args.geometry)]
    print(f"libraries: {', '.join(names)}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    results = {
        "device": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "geometry": args.geometry,
        "libraries": names,
        "quality": [],
        "agreement": [],
        "performance": [],
    }
    if "quality" in args.sections:
        print("quality")
        measure_quality(args, device, results)
    if "performance" in args.sections:
        print("speed and memory")
        measure_speed_and_memory(args, device, results)
    if "figure" in args.sections:
        print("figure")
        save_figure(args, device, args.output / "library-comparison.png")

    destination = args.output / "library-comparison.json"
    existing = json.loads(destination.read_text()) if destination.exists() else {}
    for key, value in results.items():
        if isinstance(value, list) and not value and key in existing:
            results[key] = existing[key]
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {destination}")

    if "summary" in args.sections:
        print("summary")
        save_summary_figure(results, args.output / "library-summary.png")


if __name__ == "__main__":
    main()
