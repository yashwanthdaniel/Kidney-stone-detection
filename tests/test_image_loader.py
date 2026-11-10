"""
Unit tests for image loader and corrupt file handling.
"""
import pytest
from pathlib import Path
import tempfile
from PIL import Image
import io

from backend.services.image_loader import load_image, is_dicom_file


def test_load_valid_rgb_image():
    """Verify loading valid RGB image and metadata extraction."""
    img = Image.new("RGB", (64, 48), color=(100, 150, 200))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    loaded, meta = load_image(buffer)
    assert loaded.size == (64, 48)
    assert loaded.mode == "RGB"
    assert meta["is_dicom"] is False


def test_corrupt_image_handling():
    """Verify corrupted images raise descriptive ValueError."""
    corrupt_bytes = b"NOT_A_VALID_IMAGE_HEADER_12345"
    with pytest.raises(ValueError, match="Failed to read image"):
        load_image(corrupt_bytes)


def test_dicom_magic_detection():
    """Verify DICOM detection logic."""
    assert not is_dicom_file("sample.jpg")
    assert is_dicom_file("sample.dcm")
    assert is_dicom_file("sample.dicom")
