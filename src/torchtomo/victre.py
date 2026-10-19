"""TCIA VICTRE digital breast tomosynthesis: published geometry and projection conversion.

VICTRE simulates a Siemens Mammomat Inspiration with MC-GPU: 25 views from -25 to
+25 degrees in instance-number order, source-to-detector 650 mm, source-to-axis
600 mm, and a stationary 3000 x 1500 detector of 0.085 mm pixels. The beam is
collimated to one side, so the source sits above the chest-wall edge of the
detector, not above its centre.

Coordinates follow the ASTRA ``cone_vec`` frame of the BIAI Lab DBT 2026v1
dataset: detector rows increase along tube travel (``+x``) and columns from the
chest wall toward the nipple (``+y``). Binned panels are padded to 3008 x 1536
native pixels, which is 752 x 384 after 4 x 4 binning, and volumes span
1632 x 1344 native pixels, which is 408 x 336. The per-density placement
(``NATIVE_ROWS``, ``OFFX_C``, ``DELTA``) puts VICTRE's own reconstruction grid on
that frame; it is not stored in the DICOM. Slices are 1 mm.

Stored projection pixels are rescaled detector counts,
``8 * (50 + (electrons - 5200) * 0.001469)``, not line integrals.
:func:`view_to_projection` reproduces the dataset's conversion: subtract the
400 DN pedestal, take a per-view air level from a ring around the breast, bin
before the log, and exclude collimator edges. It needs ``scipy`` and the DICOM
readers need ``pydicom``; both are in the ``tcia`` extra.
"""

import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dbt import BreastTomosynthesis, arc_angles

NATIVE_PIXEL_MM = 0.085
SOURCE_TO_AXIS_MM = 600.0
SOURCE_TO_DETECTOR_MM = 650.0
N_VIEWS = 25
HALF_ANGLE_DEG = 25.0
PANEL_NATIVE = (3008, 1536)
VOLUME_NATIVE = (1632, 1344)
SLICE_MM = 1.0
PEDESTAL_DN = 400.0
EXPOSED_MIN_DN = 800.0
FIELD_MARGIN_PX = 16
BREAST_FRACTION = 0.85
RING_PX = (30, 120)
DENSITIES = ("dense", "hetero", "scattered", "fatty")
DENSITY_SLICES = {"dense": 38, "hetero": 47, "scattered": 57, "fatty": 62}
NATIVE_ROWS = {"dense": 1130, "hetero": 1148, "scattered": 1421, "fatty": 1624}
OFFX_C = -0.203
DELTA = {"dense": 19.765, "hetero": 20.242, "scattered": 20.091, "fatty": 20.807}
_RESCALING = re.compile(r"8\s*\*\s*\(\s*50\s*\+\s*\(\s*pixel\s*-\s*5200\s*\)\s*\*\s*0\.001469\s*\)", re.IGNORECASE)


def binned_shape(native_shape: tuple[int, int], bin_factor: int) -> tuple[int, int]:
    if int(bin_factor) != bin_factor or bin_factor < 1 or any(size % bin_factor for size in native_shape):
        raise ValueError(f"bin_factor must divide {native_shape}")
    return (native_shape[0] // bin_factor, native_shape[1] // bin_factor)


def victre_angles(*, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> torch.Tensor:
    """Published view angles in radians, in instance-number order."""
    half = math.radians(HALF_ANGLE_DEG)
    return arc_angles(N_VIEWS, 2 * half, -half, dtype=dtype, device=device)


def chest_wall_y(bin_factor: int = 4) -> float:
    """Outer edge of detector column 0, the chest-wall side, in millimetres."""
    columns = binned_shape(PANEL_NATIVE, bin_factor)[1]
    return -columns * NATIVE_PIXEL_MM * bin_factor / 2


def volume_box(density: str, slices: int | None = None) -> tuple[float, float, float, float, float, float]:
    """Outer edges ``(x0, x1, y0, y1, z0, z1)`` of the reconstruction grid, in millimetres.

    The box does not depend on binning. ``x`` follows volume rows, ``y`` columns,
    and ``y0`` is the chest wall.
    """
    if density not in DENSITIES:
        raise ValueError(f"density must be one of {DENSITIES}")
    slices = DENSITY_SLICES[density] if slices is None else slices
    if int(slices) != slices or slices < 1:
        raise ValueError("slices must be a positive integer")
    width_x, width_y = (size * NATIVE_PIXEL_MM for size in VOLUME_NATIVE)
    center_x = (width_x - NATIVE_ROWS[density] * NATIVE_PIXEL_MM) / 2 + OFFX_C
    y0 = chest_wall_y()
    center_z = -(SOURCE_TO_DETECTOR_MM - SOURCE_TO_AXIS_MM) + slices * SLICE_MM / 2 + DELTA[density]
    half_z = slices * SLICE_MM / 2
    return (center_x - width_x / 2, center_x + width_x / 2, y0, y0 + width_y, center_z - half_z, center_z + half_z)


def victre_projector(
    density: str,
    slices: int | None = None,
    bin_factor: int = 4,
    *,
    views: list[int] | torch.Tensor | None = None,
    depth_margin: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    **options,
) -> BreastTomosynthesis:
    """Projector for one VICTRE density on the dataset grid at ``bin_factor``.

    Volumes are ``[B, 1, slices + 2 * depth_margin, columns, rows]``, which is
    ``(Z, Y, X)``; transpose the last two axes for the dataset's
    ``(slices, rows, columns)`` layout. Projections are
    ``[B, 1, views, rows, columns]``. ``views`` selects a subset of the 25 views,
    and ``depth_margin`` adds 1 mm slices above and below the box. Remaining
    keyword arguments, such as ``n_samples`` and ``ray_chunk``, go to
    :class:`~torchtomo.dbt.BreastTomosynthesis` and can override the geometry.
    """
    slices = DENSITY_SLICES[density] if slices is None else slices
    x0, x1, y0, y1, z0, z1 = volume_box(density, slices)
    rows, columns = binned_shape(VOLUME_NATIVE, bin_factor)
    pixel = NATIVE_PIXEL_MM * bin_factor
    angles = victre_angles(dtype=dtype, device=device)
    if views is not None:
        angles = angles[torch.as_tensor(views, device=angles.device)]
    geometry = dict(
        angles=angles,
        detector_shape=binned_shape(PANEL_NATIVE, bin_factor),
        pixel_size=pixel,
        source_to_detector=SOURCE_TO_DETECTOR_MM,
        source_to_axis=SOURCE_TO_AXIS_MM,
        source_y=chest_wall_y(bin_factor),
        voxel_spacing=(SLICE_MM, pixel, pixel),
        volume_center=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
    )
    geometry.update(options)
    return BreastTomosynthesis((slices + 2 * depth_margin, columns, rows), dtype=dtype, device=device, **geometry)


def read_stack(folder: Path) -> list:
    """Every DICOM file below ``folder``, sorted by instance number."""
    import pydicom

    datasets = [pydicom.dcmread(path) for path in Path(folder).rglob("*.dcm")]
    if not datasets:
        raise FileNotFoundError(f"no DICOM files below {folder}")
    datasets.sort(key=lambda dataset: int(dataset.InstanceNumber))
    instances = [int(dataset.InstanceNumber) for dataset in datasets]
    if len(set(instances)) != len(instances):
        raise ValueError(f"duplicate instance numbers in {folder}")
    return datasets


def read_projections(folder: Path) -> np.ndarray:
    """The 25 for-processing views of one phantom as stored DN, ``[views, 3000, 1500]``."""
    datasets = read_stack(folder)
    if len(datasets) != N_VIEWS:
        raise ValueError(f"expected {N_VIEWS} projection views, found {len(datasets)}")
    for dataset in datasets:
        if int(dataset.BitsAllocated) != 16 or int(dataset.PixelRepresentation) != 0:
            raise ValueError("VICTRE projections are stored as unsigned 16-bit pixels")
        if _RESCALING.search(str(getattr(dataset, "DerivationDescription", ""))) is None:
            raise ValueError("projection is missing the published detector-count rescaling")
        if abs(float(dataset.DistanceSourceToDetector) - SOURCE_TO_DETECTOR_MM) > 1:
            raise ValueError("projection does not use the published 650 mm source-to-detector distance")
        spacing = getattr(dataset, "ImagerPixelSpacing", None) or dataset.PixelSpacing
        if any(abs(float(value) - NATIVE_PIXEL_MM) > 1e-6 for value in spacing):
            raise ValueError(f"expected {NATIVE_PIXEL_MM} mm pixels, found {list(spacing)}")
    return np.stack([dataset.pixel_array.astype(np.float32) for dataset in datasets])


def read_reference(folder: Path) -> np.ndarray:
    """VICTRE's own reconstruction, ``[slices, rows, columns]`` at 0.085 mm, arbitrary scale."""
    return np.stack([dataset.pixel_array.astype(np.float32) for dataset in read_stack(folder)])


def density_from_comments(comments: str) -> str:
    text = comments.lower()
    if "heterogen" in text:
        return "hetero"
    for density in ("scattered", "fatty", "dense"):
        if re.search(rf"\b{density}\b", text):
            return density
    raise ValueError(f"VICTRE patient comments do not name a density: {comments!r}")


def bin_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Average ``factor x factor`` blocks of the last two axes, dropping incomplete blocks."""
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
    leading = tensor.shape[:-2]
    pooled = F.avg_pool2d(tensor.reshape(-1, 1, *tensor.shape[-2:]), factor)
    return pooled.reshape(*leading, *pooled.shape[-2:]).numpy()


def fit_to(stack: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Crop or zero-pad the last two axes at their far ends."""
    rows, columns = shape
    stack = stack[..., :rows, :columns]
    padding = [(0, 0)] * (stack.ndim - 2) + [(0, rows - stack.shape[-2]), (0, columns - stack.shape[-1])]
    return np.pad(stack, padding)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return ndimage.binary_fill_holes(labels == np.argmax(sizes) + 1)


def view_to_projection(view: np.ndarray, bin_factor: int = 4) -> tuple[np.ndarray, np.ndarray, float]:
    """Binned line integrals, valid-ray mask, and air level in DN above the pedestal.

    Line integrals are ``log(I0 / I)`` inside the breast footprint and zero
    elsewhere. ``I0`` is the median signal 30 to 120 native pixels outside the
    breast. Valid rays are binned pixels fully inside the exposed field, eroded by
    16 native pixels. Outputs are not padded; use :func:`fit_to` with
    :func:`binned_shape` of ``PANEL_NATIVE``.
    """
    from scipy import ndimage

    view = np.asarray(view, dtype=np.float32)
    exposed = ndimage.binary_erosion(ndimage.binary_fill_holes(view > EXPOSED_MIN_DN), iterations=FIELD_MARGIN_PX)
    if not exposed.any():
        raise ValueError("projection has no exposed detector field")
    air_level = np.percentile(view[exposed], 90)
    breast = _largest_component(exposed & (view < BREAST_FRACTION * air_level))
    distance = ndimage.distance_transform_edt(~breast)
    ring = (distance > RING_PX[0]) & (distance < RING_PX[1]) & exposed & (view > BREAST_FRACTION * air_level)
    if not ring.any():
        raise ValueError("could not find air around the breast for the flat field")
    signal = np.clip(view - PEDESTAL_DN, 1.0, None)
    i0 = float(np.median(signal[ring]))
    attenuation = np.log(i0 / np.clip(bin_mean(signal, bin_factor), 1.0, None))
    footprint = bin_mean((distance <= 10).astype(np.float32), bin_factor) > 0
    valid = bin_mean(exposed.astype(np.float32), bin_factor) > 0.999
    attenuation = np.where(footprint & valid, np.clip(attenuation, 0, None), 0.0)
    return attenuation.astype(np.float32), valid, i0


def convert_projections(views: np.ndarray, bin_factor: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padded line integrals ``[views, rows, columns]``, valid mask, and per-view air levels."""
    converted = [view_to_projection(view, bin_factor) for view in views]
    shape = binned_shape(PANEL_NATIVE, bin_factor)
    projections = fit_to(np.stack([item[0] for item in converted]), shape)
    valid = fit_to(np.stack([item[1] for item in converted]), shape)
    return projections, valid, np.array([item[2] for item in converted], dtype=np.float32)


def reference_on_grid(reference: np.ndarray, bin_factor: int = 4) -> np.ndarray:
    """VICTRE's reconstruction binned onto the dataset grid, ``[slices, rows, columns]``."""
    return fit_to(bin_mean(reference, bin_factor), binned_shape(VOLUME_NATIVE, bin_factor))
