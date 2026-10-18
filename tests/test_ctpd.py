"""CT-PD byte order, storage order, physical coordinates, and focal-spot offsets."""

import struct

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from torchtomo.ctpd import View, read_projection  # noqa: E402


def dataset():
    data = pydicom.Dataset()
    data.file_meta = pydicom.FileMetaDataset()
    data.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    data.file_meta.MediaStorageSOPClassUID = pydicom.uid.RawDataStorage
    data.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    data.SOPClassUID = data.file_meta.MediaStorageSOPClassUID
    data.SOPInstanceUID = data.file_meta.MediaStorageSOPInstanceUID
    data.InstanceNumber = 17
    data.Rows, data.Columns = 3, 2
    data.BitsAllocated = data.BitsStored = 16
    data.HighBit = 15
    data.PixelRepresentation = 0
    data.RescaleSlope, data.RescaleIntercept = 0.25, -0.5
    for tag, value in {
        (0x7029, 0x1010): 2,
        (0x7029, 0x1011): 3,
        (0x7033, 0x1013): 100,
        (0x7033, 0x1061): 1,
    }.items():
        data.add_new(tag, "UN", struct.pack("<H", value))
    for tag, value in {
        (0x7029, 0x1002): 2.0,
        (0x7029, 0x1006): 3.0,
        (0x7031, 0x1001): 0.0,
        (0x7031, 0x1002): 5.0,
        (0x7031, 0x1003): 10.0,
        (0x7031, 0x1031): 20.0,
        (0x7033, 0x100B): 0.0,
        (0x7033, 0x100C): 0.0,
        (0x7033, 0x100D): 0.0,
    }.items():
        data.add_new(tag, "UN", struct.pack("<f", value))
    data.add_new((0x7031, 0x1033), "UN", struct.pack("<2f", 2.0, 1.5))
    data.add_new((0x7029, 0x100B), "UN", b"CYLINDRICAL ")
    data.add_new((0x7037, 0x100A), "UN", b"FANBEAM ")
    data.add_new((0x7041, 0x1001), "UN", b"0.0186")
    data.PixelData = np.arange(6, dtype="<u2").tobytes()
    return data


def test_storage_order_and_rescale(tmp_path):
    data = dataset()
    path = tmp_path / "view.dcm"
    pydicom.dcmwrite(path, data, enforce_file_format=True)
    view, actual = read_projection(path)
    assert view.instance == 17
    assert (view.rows, view.columns) == (2, 3)
    expected = np.array([[0, 2, 4], [1, 3, 5]]) * 0.25 - 0.5
    np.testing.assert_allclose(actual, expected)


def test_manual_geometry_and_detector_stays_fixed_under_source_motion():
    data = dataset()
    view = View.from_dataset(data)
    source, detector = view.positions(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(source, [0, 10, 5])
    np.testing.assert_allclose(detector[:, 1], [[0, -10, 6.5], [0, -10, 3.5]])
    np.testing.assert_allclose(np.linalg.norm(detector[0, :, :2] - source[:2], axis=-1), 20)
    assert detector[0, 0, 0] < 0 < detector[0, 2, 0]
    data[(0x7033, 0x100B)].value = struct.pack("<f", np.pi / 2)
    data[(0x7033, 0x100C)].value = struct.pack("<f", 2)
    data[(0x7033, 0x100D)].value = struct.pack("<f", 1)
    shifted_source, shifted_detector = View.from_dataset(data).positions(np.array([0.0, 1.0]), np.array([0, 1, 2]))
    np.testing.assert_allclose(shifted_source, [-11, 0, 7], atol=1e-6)
    np.testing.assert_array_equal(shifted_detector, detector)


@pytest.mark.parametrize("problem", ["shape", "dimensions", "spectra", "pixels"])
def test_unsupported_or_inconsistent_data_rejected(problem):
    data = dataset()
    if problem == "shape":
        data[(0x7029, 0x100B)].value = b"SPHERICAL "
    elif problem == "dimensions":
        data.Rows = 4
    elif problem == "spectra":
        data[(0x7033, 0x1061)].value = struct.pack("<H", 2)
    else:
        data.PixelRepresentation = 1
    with pytest.raises(ValueError):
        View.from_dataset(data)
