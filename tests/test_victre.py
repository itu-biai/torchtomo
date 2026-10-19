"""Published VICTRE geometry, the dataset grid, and detector-count conversion."""

import math

import numpy as np
import pytest
import torch

from torchtomo.victre import (
    DELTA,
    NATIVE_ROWS,
    OFFX_C,
    PEDESTAL_DN,
    binned_shape,
    chest_wall_y,
    convert_projections,
    density_from_comments,
    read_projections,
    reference_on_grid,
    victre_angles,
    victre_projector,
    view_to_projection,
    volume_box,
)


def astra_cone_vectors(bin_factor: int = 4) -> np.ndarray:
    """``cone_vectors()`` of the BIAI Lab DBT 2026v1 generator."""
    pixel = 0.085 * bin_factor
    angles = np.radians(np.linspace(-25, 25, 25))
    vectors = np.zeros((25, 12))
    vectors[:, 0] = 600 * np.sin(angles)
    vectors[:, 1] = -(1536 // bin_factor) * pixel / 2
    vectors[:, 2] = 600 * np.cos(angles)
    vectors[:, 5] = -50
    vectors[:, 7] = pixel
    vectors[:, 9] = pixel
    return vectors


@pytest.mark.parametrize("bin_factor", [4, 8])
def test_positions_match_the_dataset_astra_vectors(bin_factor):
    operator = victre_projector("scattered", bin_factor=bin_factor, dtype=torch.float64)
    vectors = torch.as_tensor(astra_cone_vectors(bin_factor))
    sources, detectors = operator.source_positions, operator.detector_positions
    rows, columns = binned_shape((3008, 1536), bin_factor)
    assert detectors.shape == (25, rows, columns, 3)
    torch.testing.assert_close(sources, vectors[:, 0:3])
    torch.testing.assert_close(detectors.mean(dim=(1, 2)), vectors[:, 3:6])
    torch.testing.assert_close(detectors[:, 0, 1] - detectors[:, 0, 0], vectors[:, 6:9])
    torch.testing.assert_close(detectors[:, 1, 0] - detectors[:, 0, 0], vectors[:, 9:12])


def test_chest_wall_and_volume_box_follow_the_generator():
    assert chest_wall_y(4) == pytest.approx(-384 * 0.34 / 2)
    assert chest_wall_y(8) == pytest.approx(chest_wall_y(4))
    center_x = (408 * 0.34 - NATIVE_ROWS["scattered"] * 0.085) / 2 + OFFX_C
    y0 = -384 * 0.34 / 2
    center_z = -50 + 57 / 2 + DELTA["scattered"]
    expected = [center_x - 69.36, center_x + 69.36, y0, y0 + 114.24, center_z - 28.5, center_z + 28.5]
    np.testing.assert_allclose(volume_box("scattered"), expected)


def test_projector_grid_views_and_depth_margin():
    operator = victre_projector("dense", bin_factor=16, views=[0, 12, 24], depth_margin=2, n_samples=8)
    assert operator.operator.volume_shape == (38 + 4, 84, 102)
    assert operator.operator.projection_shape == (3, 188, 96)
    torch.testing.assert_close(operator.angles, victre_angles()[[0, 12, 24]])
    spacing = operator.operator.spacing_xyz
    torch.testing.assert_close(spacing, spacing.new_tensor([1.36, 1.36, 1.0]))
    x0, x1, y0, y1, z0, z1 = volume_box("dense")
    torch.testing.assert_close(
        operator.operator.volume_center, torch.tensor([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    )
    volume = torch.ones(1, 1, 42, 84, 102)
    assert operator(volume).amax() > 0


def test_density_names_do_not_confuse_heterogeneous_with_dense():
    assert density_from_comments("Scattered breast type") == "scattered"
    assert density_from_comments("Heterogeneously dense") == "hetero"
    assert density_from_comments("Dense breast type") == "dense"
    assert density_from_comments("fatty") == "fatty"
    with pytest.raises(ValueError):
        density_from_comments("unknown")


def synthetic_view() -> tuple[np.ndarray, float, float]:
    """Exposed field with an air level, a uniform breast against the chest wall, and collimated borders."""
    air_signal, breast_signal = 3000.0, 1200.0
    view = np.full((400, 320), 300.0, dtype=np.float32)
    view[20:380, 0:300] = PEDESTAL_DN + air_signal
    view[120:280, 0:120] = PEDESTAL_DN + breast_signal
    return view, air_signal, breast_signal


def test_view_conversion_is_log_of_air_over_signal_inside_the_breast():
    pytest.importorskip("scipy")
    view, air_signal, breast_signal = synthetic_view()
    attenuation, valid, i0 = view_to_projection(view, bin_factor=4)
    assert attenuation.shape == valid.shape == (100, 80)
    assert i0 == pytest.approx(air_signal)
    assert attenuation[50, 10] == pytest.approx(math.log(air_signal / breast_signal), rel=1e-5)
    assert attenuation[10, 75] == 0 and attenuation[50, 60] == 0
    assert valid[50, 40] and not valid[2, 40] and not valid[50, 79]
    projections, padded_valid, air = convert_projections(np.stack([view, view]), bin_factor=16)
    assert projections.shape == padded_valid.shape == (2, 188, 96)
    assert not padded_valid[:, 25:].any() and air.shape == (2,)


def test_reference_is_binned_and_padded_onto_the_dataset_grid():
    reference = np.ones((3, 1421, 1024), dtype=np.float32)
    grid = reference_on_grid(reference, bin_factor=4)
    assert grid.shape == (3, 408, 336)
    assert grid[:, :355, :256].min() == 1 and grid[:, 355:].max() == 0 and grid[:, :, 256:].max() == 0


def write_view(path, instance: int, description: str) -> None:
    pydicom = pytest.importorskip("pydicom")
    data = pydicom.Dataset()
    data.file_meta = pydicom.FileMetaDataset()
    data.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    data.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2.1"
    data.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    data.SOPClassUID = data.file_meta.MediaStorageSOPClassUID
    data.SOPInstanceUID = data.file_meta.MediaStorageSOPInstanceUID
    data.Rows, data.Columns = 2, 3
    data.BitsAllocated = data.BitsStored = 16
    data.HighBit = 15
    data.SamplesPerPixel = 1
    data.PhotometricInterpretation = "MONOCHROME2"
    data.PixelRepresentation = 0
    data.InstanceNumber = instance
    data.ImagerPixelSpacing = [0.085, 0.085]
    data.DistanceSourceToDetector = 650
    data.DerivationDescription = description
    data.PixelData = np.full(6, instance, dtype="<u2").tobytes()
    pydicom.dcmwrite(path, data, enforce_file_format=True)


def test_reader_sorts_by_instance_and_requires_the_published_rescaling(tmp_path):
    published = "rescaled raw data = 8*(50+ (pixel-5200)*0.001469)"
    for instance in range(25, 0, -1):
        write_view(tmp_path / f"view{instance:02d}.dcm", instance, published)
    views = read_projections(tmp_path)
    assert views.shape == (25, 2, 3)
    np.testing.assert_array_equal(views[:, 0, 0], np.arange(1, 26))
    write_view(tmp_path / "view01.dcm", 1, "not the published mapping")
    with pytest.raises(ValueError, match="rescaling"):
        read_projections(tmp_path)
