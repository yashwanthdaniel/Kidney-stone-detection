"""
Unit tests for YOLO detection fallback, SQLite database, PDF reports, and measurements.
"""
import pytest
from pathlib import Path
import tempfile
from PIL import Image

from backend.services.detection import run_yolo_detection, is_detector_available
from backend.services.measurement import calculate_region_measurement
from backend.database.db_manager import save_examination, get_examination_by_id
from backend.reporting.pdf_generator import generate_pdf_report


def test_yolo_missing_checkpoint_fallback():
    """Verify missing detector returns transparent status without inventing bounding boxes."""
    dummy_img = Image.new("RGB", (100, 100), color="black")
    res = run_yolo_detection(dummy_img)

    # Should not crash and should report available=False when no checkpoint exists
    if not is_detector_available():
        assert res["available"] is False
        assert len(res["detections"]) == 0
        assert "not trained or unavailable" in res["status"]


def test_measurement_calibrated_vs_uncalibrated():
    """Verify physical millimeter conversion when PixelSpacing is present vs absent."""
    # Calibrated case (DICOM with 0.5mm pixel spacing)
    calibrated = calculate_region_measurement(
        width_px=20.0,
        height_px=10.0,
        pixel_spacing_mm=[0.5, 0.5],
        slice_thickness_mm=1.0,
    )
    assert calibrated["calibrated"] is True
    assert calibrated["width_mm"] == 10.0
    assert calibrated["height_mm"] == 5.0
    assert calibrated["max_dimension_mm"] == 10.0

    # Uncalibrated case (Standard PNG/JPG without pixel spacing)
    uncalibrated = calculate_region_measurement(
        width_px=20.0,
        height_px=10.0,
        pixel_spacing_mm=None,
    )
    assert uncalibrated["calibrated"] is False
    assert uncalibrated["width_mm"] is None
    assert "Uncalibrated 2D ROI" in uncalibrated["measurement_status"]


def test_database_and_pdf_report_lifecycle():
    """Verify saving to SQLite and generating downloadable PDF report."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        exam_id = "TEST_EXAM_999"
        mock_preds = {
            "mobilenet_v2": {
                "available": True,
                "predicted_class": "Stone",
                "confidence": 0.94,
                "stone_probability": 0.94,
                "non_stone_probability": 0.06,
                "inference_time_ms": 12.5,
            }
        }
        mock_ens = {
            "available": True,
            "predicted_class": "Stone",
            "confidence": 0.94,
            "stone_probability": 0.94,
            "non_stone_probability": 0.06,
            "contributing_models": ["mobilenet_v2"],
            "weights": {"mobilenet_v2": 1.0},
        }

        # Save to SQLite
        save_examination(
            exam_id=exam_id,
            predictions=mock_preds,
            ensemble=mock_ens,
            filename="synthetic_test.png",
        )

        record = get_examination_by_id(exam_id)
        assert record is not None
        assert record["id"] == exam_id
        assert record["predicted_class"] == "Stone"

        # Generate PDF report
        pdf_out = Path(tmp_dir) / "test_report.pdf"
        generated_path = generate_pdf_report(
            exam_id=exam_id,
            predictions=mock_preds,
            ensemble=mock_ens,
            output_path=pdf_out,
        )

        assert generated_path.exists()
        assert generated_path.stat().st_size > 1000
