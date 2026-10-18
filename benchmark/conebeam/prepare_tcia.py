"""Prepare one rotation of measured CT-PD rays and the corresponding vendor CT volume."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as functional

from torchtomo.ctpd import View, read_projection


def header(path: Path) -> tuple[Path, View]:
    return path, View.from_dataset(pydicom.dcmread(path, stop_before_pixels=True))


def prepare(root: Path, output: Path, view_stride: int = 4) -> None:
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted((root / "full_proj").glob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"no complete projection DICOMs in {root / 'full_proj'}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        headers = sorted(pool.map(header, paths), key=lambda item: item[1].instance)
    instances = np.array([view.instance for _, view in headers])
    if not np.all(np.diff(instances) == 1):
        raise ValueError("projection instances are duplicated or missing")
    datasets = [pydicom.dcmread(path) for path in sorted((root / "full_img").glob("*.dcm"))]
    datasets.sort(key=lambda dataset: float(dataset.ImagePositionPatient[2]))
    if len(datasets) < 3:
        raise ValueError("at least three vendor CT slices are required")
    for dataset in datasets:
        if not np.allclose(dataset.ImageOrientationPatient, [1, 0, 0, 0, 1, 0]):
            raise ValueError("this pilot requires axial vendor CT images in the DICOM patient frame")
        if not np.allclose(dataset.PixelSpacing, datasets[0].PixelSpacing):
            raise ValueError("vendor CT pixel spacing changes within the series")
    reference_z = np.array([float(dataset.ImagePositionPatient[2]) for dataset in datasets])
    if not np.allclose(np.diff(reference_z), np.diff(reference_z)[0]):
        raise ValueError("vendor CT slices must be regularly spaced")
    target_z = float(np.median(reference_z))
    middle = min(range(len(headers)), key=lambda index: abs(headers[index][1].z - target_z))
    per_rotation = headers[middle][1].views_per_rotation
    start = middle - per_rotation // 2
    if start < 0 or start + per_rotation > len(headers):
        raise ValueError("the central vendor CT slice lacks a complete rotation")
    selected = headers[start : start + per_rotation : view_stride]
    if len(selected) < 32 or view_stride < 1:
        raise ValueError("at least 32 views are required")
    raw_angles = np.unwrap([view.angle for _, view in selected])
    if abs(abs(raw_angles[-1] - raw_angles[0]) - 2 * np.pi) > 4 * np.pi * view_stride / per_rotation:
        raise ValueError("selected views do not cover one rotation")
    first = selected[0][1]
    rows = np.arange(0, first.rows, 4)
    columns = np.arange(0, first.columns, 4)
    sources, detectors, projections = [], [], []
    for path, expected in selected:
        view, pixels = read_projection(path)
        if view != expected:
            raise ValueError("projection header changed after indexing")
        source, detector = view.positions(rows, columns)
        sources.append(source)
        detectors.append(detector)
        projections.append(pixels[np.ix_(rows, columns)])
    original_hu = np.stack(
        [
            dataset.pixel_array.astype(np.float32) * float(dataset.RescaleSlope) + float(dataset.RescaleIntercept)
            for dataset in datasets
        ]
    )
    image_spacing = np.array(datasets[0].PixelSpacing, dtype=np.float64)
    reference_hu = functional.interpolate(torch.from_numpy(original_hu)[:, None], size=(128, 128), mode="area")
    reference_hu = reference_hu[:, 0].numpy()
    reference_spacing = image_spacing * np.array(original_hu.shape[-2:]) / 128
    position = np.array(datasets[0].ImagePositionPatient, dtype=np.float64)
    center_xy = position[:2] + image_spacing[::-1] * (np.array(original_hu.shape[-2:][::-1]) - 1) / 2
    metadata = {
        "patient": root.name,
        "manufacturer": str(datasets[0].Manufacturer),
        "patient_position": str(datasets[0].PatientPosition),
        "complete_projection_files": len(paths),
        "selected_instances": [view.instance for _, view in selected],
        "first_view": asdict(first),
        "target_z": target_z,
        "source_z_range": [min(view.z for _, view in selected), max(view.z for _, view in selected)],
        "detector_rows": rows.tolist(),
        "detector_columns": columns.tolist(),
        "reference_thickness": float(datasets[0].SliceThickness),
        "reference_spacing_zyx": [float(np.diff(reference_z)[0]), *reference_spacing.tolist()],
        "reference_center_xy": center_xy.tolist(),
        "reference_z_range": [float(reference_z[0]), float(reference_z[-1])],
        "frame": "CT-PD manual x=-rho*sin(phi), y=rho*cos(phi), z=z; increasing detector rows decrease z",
        "split": "alternating selected views: even for training and frame selection, odd held out",
    }
    np.savez_compressed(
        output / "input.npz",
        sources=np.asarray(sources, dtype=np.float32),
        detectors=np.asarray(detectors, dtype=np.float32),
        projections=np.asarray(projections, dtype=np.float32),
        reference_hu=reference_hu,
        reference_z=reference_z,
        metadata=np.array(json.dumps(metadata)),
    )
    (output / "input_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view-stride", type=int, default=4)
    args = parser.parse_args()
    prepare(args.root, args.output, args.view_stride)
