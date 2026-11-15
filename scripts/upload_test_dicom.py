"""Convert one RSNA PNG to a DICOM Secondary Capture and upload it to Orthanc.

Usage: python scripts/upload_test_dicom.py path/to/image.png
For education only, not for clinical use.
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import requests
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (ExplicitVRLittleEndian, SecondaryCaptureImageStorage,
                         generate_uid)

ORTHANC_URL = "http://localhost:8042/instances"
OUT_DIR = Path("data/processed")


def build_dicom(png_path):
    img = Image.open(png_path).convert("L")  # 8-bit grayscale
    pixels = np.array(img, dtype=np.uint8)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    now = datetime.now()
    ds.StudyDate = ds.SeriesDate = ds.ContentDate = now.strftime("%Y%m%d")
    ds.StudyTime = ds.SeriesTime = ds.ContentTime = now.strftime("%H%M%S")
    ds.PatientID = "RSNA_TEST_ONLY"
    ds.PatientName = "RSNA^TEST^ONLY"
    ds.Modality = "OT"
    ds.ConversionType = "WSD"
    ds.StudyID = "1"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    return ds


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/upload_test_dicom.py <image.png>")
    png_path = Path(sys.argv[1])
    if not png_path.is_file():
        sys.exit(f"Error: PNG not found: {png_path}")

    try:
        ds = build_dicom(png_path)
    except Exception as e:
        sys.exit(f"Error: could not read/convert image: {e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dcm_path = OUT_DIR / f"{ds.SOPInstanceUID}.dcm"
    ds.save_as(dcm_path, enforce_file_format=True)
    print(f"Saved DICOM: {dcm_path}")

    try:
        r = requests.post(ORTHANC_URL, data=dcm_path.read_bytes(),
                          headers={"Content-Type": "application/dicom"}, timeout=30)
    except requests.exceptions.ConnectionError:
        sys.exit(f"Error: cannot connect to Orthanc at {ORTHANC_URL}. Is it running?")
    except requests.exceptions.RequestException as e:
        sys.exit(f"Error: upload failed: {e}")

    if r.status_code != 200:
        sys.exit(f"Error: Orthanc returned HTTP {r.status_code}: {r.text[:300]}")
    info = r.json()
    print(f"Upload status: {info.get('Status')} (HTTP {r.status_code})")
    print(f"Orthanc instance ID: {info.get('ID')}")


if __name__ == "__main__":
    main()
