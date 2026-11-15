import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
from fastapi import HTTPException
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.inference_api import main  # noqa: E402

ROWS, COLS = 4, 6


def make_dicom(pixels, photometric="MONOCHROME2", frames=None, **tags):
    """Build a small synthetic 16-bit DICOM and return its bytes."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = ROWS, COLS
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    if frames:
        ds.NumberOfFrames = frames
    for k, v in tags.items():
        setattr(ds, k, v)
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf)
    return buf.getvalue()


def gradient():
    return np.arange(ROWS * COLS).reshape(ROWS, COLS) * 100


def test_monochrome2_returns_rgb_image():
    img = main.dicom_to_image(make_dicom(gradient()))
    assert img.mode == "RGB"
    assert img.size == (COLS, ROWS)
    arr = np.array(img)
    assert arr.min() == 0 and arr.max() == 255
    assert arr[0, 0, 0] == 0 and arr[-1, -1, 0] == 255  # low value is dark


def test_monochrome1_is_inverted():
    img = main.dicom_to_image(make_dicom(gradient(), photometric="MONOCHROME1"))
    assert img.mode == "RGB"
    assert img.size == (COLS, ROWS)
    arr = np.array(img)
    assert arr[0, 0, 0] == 255 and arr[-1, -1, 0] == 0  # low value is bright


def test_rescale_slope_and_intercept_applied():
    # Negative slope flips the ordering, so the first pixel becomes the brightest
    img = main.dicom_to_image(make_dicom(gradient(), RescaleSlope=-1, RescaleIntercept=1000))
    arr = np.array(img)
    assert arr[0, 0, 0] == 255 and arr[-1, -1, 0] == 0


def test_unsupported_photometric_rejected():
    with pytest.raises(HTTPException) as e:
        main.dicom_to_image(make_dicom(gradient(), photometric="PALETTE COLOR"))
    assert e.value.status_code == 422


def test_multi_frame_rejected():
    data = make_dicom(np.zeros((2, ROWS, COLS)), frames=2)
    with pytest.raises(HTTPException) as e:
        main.dicom_to_image(data)
    assert e.value.status_code == 422


def test_invalid_bytes_rejected():
    with pytest.raises(HTTPException) as e:
        main.dicom_to_image(b"not a dicom file")
    assert e.value.status_code == 422


def test_unreadable_pixel_data_rejected():
    data = make_dicom(gradient())
    ds = pydicom.dcmread(io.BytesIO(data))
    del ds.PixelData  # valid header, no pixel data
    buf = io.BytesIO()
    ds.save_as(buf)
    with pytest.raises(HTTPException) as e:
        main.dicom_to_image(buf.getvalue())
    assert e.value.status_code == 422


def test_non_2d_pixel_array_rejected(monkeypatch):
    # Fake a dataset whose pixel array is 3D but has no NumberOfFrames tag
    class FakeDs(SimpleNamespace):
        def get(self, key, default=None):
            return {"PhotometricInterpretation": "MONOCHROME2"}.get(key, default)

    monkeypatch.setattr(main.pydicom, "dcmread", lambda _: FakeDs(pixel_array=np.zeros((2, ROWS, COLS))))
    with pytest.raises(HTTPException) as e:
        main.dicom_to_image(b"ignored")
    assert e.value.status_code == 422
