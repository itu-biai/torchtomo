"""DICOM-CT-PD decoding and cylindrical detector geometry, in the manual's Cartesian frame."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom


def numeric(dataset: pydicom.Dataset, tag: tuple[int, int], dtype: str, count: int = 1) -> np.ndarray:
    value = dataset[tag].value
    if isinstance(value, bytes):
        endian = "<" if dataset.file_meta.TransferSyntaxUID.is_little_endian else ">"
        value = np.frombuffer(value, dtype=endian + dtype)
    result = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if result.size != count or not np.isfinite(result).all():
        raise ValueError(f"invalid value count or nonfinite data for {tag}: expected {count}")
    return result


def text(dataset: pydicom.Dataset, tag: tuple[int, int]) -> str:
    value = dataset[tag].value
    return (value.decode("ascii") if isinstance(value, bytes) else str(value)).strip("\x00 ")


@dataclass(frozen=True)
class View:
    instance: int
    angle: float
    z: float
    radius: float
    source_detector_distance: float
    channel_spacing: float
    row_spacing: float
    central_channel: float
    central_row: float
    rows: int
    columns: int
    source_angle_shift: float
    source_z_shift: float
    source_radius_shift: float
    views_per_rotation: int
    water_attenuation: float

    @classmethod
    def from_dataset(cls, dataset: pydicom.Dataset) -> "View":
        if text(dataset, (0x7029, 0x100B)) != "CYLINDRICAL":
            raise ValueError("the CT-PD adapter currently requires a cylindrical detector")
        if text(dataset, (0x7037, 0x100A)) != "FANBEAM":
            raise ValueError("the CT-PD adapter requires a third-generation point-source scan")
        if int(numeric(dataset, (0x7033, 0x1061), "u2")[0]) != 1:
            raise ValueError("multiple spectra require separate reconstruction models")
        if int(dataset.BitsAllocated) != 16 or int(dataset.PixelRepresentation) != 0:
            raise ValueError("the CT-PD adapter currently requires unsigned 16-bit stored pixels")
        if dataset.file_meta.TransferSyntaxUID.is_compressed:
            raise ValueError("compressed CT-PD pixel storage is not supported")
        rows = int(numeric(dataset, (0x7029, 0x1010), "u2")[0])
        columns = int(numeric(dataset, (0x7029, 0x1011), "u2")[0])
        if (int(dataset.Rows), int(dataset.Columns)) != (columns, rows):
            raise ValueError("CT-PD pixel dimensions do not match channel-major detector storage")
        central = numeric(dataset, (0x7031, 0x1033), "f4", 2)
        water_value = dataset[(0x7041, 0x1001)].value
        if isinstance(water_value, bytes):
            try:
                water_value = float(water_value.decode("ascii").strip("\x00 "))
            except (UnicodeDecodeError, ValueError):
                water_value = numeric(dataset, (0x7041, 0x1001), "f4")[0]
        result = cls(
            instance=int(dataset.InstanceNumber),
            angle=float(numeric(dataset, (0x7031, 0x1001), "f4")[0]),
            z=float(numeric(dataset, (0x7031, 0x1002), "f4")[0]),
            radius=float(numeric(dataset, (0x7031, 0x1003), "f4")[0]),
            source_detector_distance=float(numeric(dataset, (0x7031, 0x1031), "f4")[0]),
            channel_spacing=float(numeric(dataset, (0x7029, 0x1002), "f4")[0]),
            row_spacing=float(numeric(dataset, (0x7029, 0x1006), "f4")[0]),
            central_channel=float(central[0]),
            central_row=float(central[1]),
            rows=rows,
            columns=columns,
            source_angle_shift=float(numeric(dataset, (0x7033, 0x100B), "f4")[0]),
            source_z_shift=float(numeric(dataset, (0x7033, 0x100C), "f4")[0]),
            source_radius_shift=float(numeric(dataset, (0x7033, 0x100D), "f4")[0]),
            views_per_rotation=int(numeric(dataset, (0x7033, 0x1013), "u2")[0]),
            water_attenuation=float(water_value),
        )
        positive = (
            result.radius,
            result.source_detector_distance,
            result.channel_spacing,
            result.row_spacing,
            result.water_attenuation,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("CT-PD lengths and water attenuation must be finite and positive")
        if rows < 1 or columns < 1 or result.views_per_rotation < 1:
            raise ValueError("CT-PD detector and angular dimensions must be positive")
        return result

    def positions(self, rows: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute physical points, preserving the fixed detector when the focal spot moves."""
        angle = self.angle
        focal = np.array([-self.radius * np.sin(angle), self.radius * np.cos(angle), self.z])
        source_angle = angle + self.source_angle_shift
        source_radius = self.radius + self.source_radius_shift
        source = np.array(
            [-source_radius * np.sin(source_angle), source_radius * np.cos(source_angle), self.z + self.source_z_shift]
        )
        inward = np.array([np.sin(angle), -np.cos(angle), 0.0])
        tangent = np.array([np.cos(angle), np.sin(angle), 0.0])
        fan = (columns - (self.central_channel - 1)) * self.channel_spacing / self.source_detector_distance
        radial = np.cos(fan)[:, None] * inward + np.sin(fan)[:, None] * tangent
        detector = focal + self.source_detector_distance * radial
        detector = np.broadcast_to(detector, (len(rows), len(columns), 3)).copy()
        detector[..., 2] -= (rows[:, None] - (self.central_row - 1)) * self.row_spacing
        return source, detector


def read_projection(path: Path) -> tuple[View, np.ndarray]:
    dataset = pydicom.dcmread(path)
    view = View.from_dataset(dataset)
    endian = "<" if dataset.file_meta.TransferSyntaxUID.is_little_endian else ">"
    values = np.frombuffer(dataset.PixelData, dtype=endian + "u2")
    if values.size != view.rows * view.columns:
        raise ValueError(f"incomplete CT-PD pixel data in {path}")
    pixels = values.reshape(view.columns, view.rows).T.astype(np.float32)
    slope, intercept = float(dataset.RescaleSlope), float(dataset.RescaleIntercept)
    if not np.isfinite(slope) or slope <= 0 or not np.isfinite(intercept):
        raise ValueError("invalid CT-PD projection rescale parameters")
    return view, pixels * slope + intercept
