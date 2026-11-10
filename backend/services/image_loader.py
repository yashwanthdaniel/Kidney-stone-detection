"""
Image Loader Service
Supports standard image formats (PNG, JPG, JPEG, BMP) and DICOM medical files.
Handles error cases, corrupt images, and metadata extraction.
"""
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import io

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def is_dicom_file(file_or_path: Union[str, Path, io.BytesIO]) -> bool:
    """Check if the given file or path is a DICOM file."""
    if isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        if p.suffix.lower() in [".dcm", ".dicom"]:
            return True
        try:
            with open(p, "rb") as f:
                f.seek(128)
                return f.read(4) == b"DICM"
        except Exception:
            return False
    elif hasattr(file_or_path, "seek") and hasattr(file_or_path, "read"):
        try:
            pos = file_or_path.tell()
            file_or_path.seek(128)
            magic = file_or_path.read(4)
            file_or_path.seek(pos)
            return magic == b"DICM"
        except Exception:
            return False
    return False


def load_dicom_image(file_or_path: Union[str, Path, io.BytesIO]) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Load a DICOM file, extract pixel array normalized to 8-bit RGB, and return metadata.
    """
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required to read DICOM files. Please install pydicom.")

    ds = pydicom.dcmread(file_or_path)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present (Hounsfield Units)
    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    hu_image = pixel_array * slope + intercept

    # Normalize to 0-255 uint8
    min_val, max_val = hu_image.min(), hu_image.max()
    if max_val > min_val:
        norm_img = ((hu_image - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        norm_img = np.zeros_like(hu_image, dtype=np.uint8)

    # Convert to 3-channel RGB PIL Image
    pil_image = Image.fromarray(norm_img).convert("RGB")

    # Extract clinical / physical metadata
    pixel_spacing = getattr(ds, "PixelSpacing", None)
    spacing_val = None
    if pixel_spacing:
        try:
            spacing_val = [float(x) for x in pixel_spacing]
        except Exception:
            spacing_val = None

    metadata: Dict[str, Any] = {
        "is_dicom": True,
        "modality": getattr(ds, "Modality", "CT"),
        "pixel_spacing_mm": spacing_val,
        "slice_thickness_mm": float(getattr(ds, "SliceThickness", 0.0)) if hasattr(ds, "SliceThickness") else None,
        "patient_id": getattr(ds, "PatientID", "ANONYMOUS"),
        "study_date": getattr(ds, "StudyDate", "UNKNOWN"),
        "rows": getattr(ds, "Rows", norm_img.shape[0]),
        "columns": getattr(ds, "Columns", norm_img.shape[1]),
        "window_center": getattr(ds, "WindowCenter", None),
        "window_width": getattr(ds, "WindowWidth", None),
        "min_hu": float(min_val),
        "max_hu": float(max_val),
    }

    return pil_image, metadata


def load_image(file_or_path: Union[str, Path, io.BytesIO, bytes]) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Load an image from file path, bytes, or file-like object.
    Supports PNG, JPG, JPEG, BMP, and DICOM formats.
    Returns (PIL.Image, metadata_dict).
    """
    metadata: Dict[str, Any] = {
        "is_dicom": False,
        "pixel_spacing_mm": None,
        "format": None,
        "mode": None,
        "size": None,
    }

    # Check for DICOM
    if is_dicom_file(file_or_path):
        return load_dicom_image(file_or_path)

    # Load standard image
    try:
        if isinstance(file_or_path, bytes):
            img = Image.open(io.BytesIO(file_or_path))
        else:
            img = Image.open(file_or_path)

        # Force load image data to catch corrupt images early
        img.load()
        metadata["format"] = img.format
        metadata["mode"] = img.mode
        metadata["size"] = img.size

        # Convert to RGB
        rgb_img = img.convert("RGB")
        return rgb_img, metadata
    except Exception as e:
        raise ValueError(f"Failed to read image: {str(e)}") from e
