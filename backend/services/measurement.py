"""
Medical Measurement Service
Calculates physical millimeter dimensions from DICOM pixel spacing where available,
or explicitly reports pixel dimensions when standard PNG/JPEG images lack calibration.
Prevents uncalibrated clinical measurement errors.
"""
from typing import Dict, Any, Optional, List, Tuple


def calculate_region_measurement(
    width_px: float,
    height_px: float,
    pixel_spacing_mm: Optional[List[float]] = None,
    slice_thickness_mm: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate 2D dimension measurements from detected bounding box or manual ROI.
    """
    area_px2 = width_px * height_px

    if pixel_spacing_mm and len(pixel_spacing_mm) >= 2:
        row_spacing = float(pixel_spacing_mm[0])
        col_spacing = float(pixel_spacing_mm[1])

        width_mm = round(width_px * col_spacing, 2)
        height_mm = round(height_px * row_spacing, 2)
        area_mm2 = round(area_px2 * row_spacing * col_spacing, 2)
        max_dim_mm = max(width_mm, height_mm)

        return {
            "calibrated": True,
            "width_px": round(width_px, 1),
            "height_px": round(height_px, 1),
            "area_px2": round(area_px2, 1),
            "pixel_spacing_mm": [row_spacing, col_spacing],
            "width_mm": width_mm,
            "height_mm": height_mm,
            "max_dimension_mm": max_dim_mm,
            "area_mm2": area_mm2,
            "slice_thickness_mm": slice_thickness_mm,
            "measurement_status": f"Calibrated physical size: {max_dim_mm} mm (2D maximum slice diameter)",
            "limitations": [
                "2D slice measurement only: Does not establish full 3D volumetric stone burden.",
                "Subject to slice orientation and partial volume effects.",
            ],
        }
    else:
        return {
            "calibrated": False,
            "width_px": round(width_px, 1),
            "height_px": round(height_px, 1),
            "area_px2": round(area_px2, 1),
            "pixel_spacing_mm": None,
            "width_mm": None,
            "height_mm": None,
            "max_dimension_mm": None,
            "area_mm2": None,
            "measurement_status": f"Uncalibrated 2D ROI: {round(width_px)} x {round(height_px)} pixels",
            "limitations": [
                "Physical millimeter scale unavailable: Standard PNG/JPEG image does not retain DICOM PixelSpacing metadata.",
                "Reporting pixel dimensions only. Do NOT make clinical decisions based on uncalibrated pixel sizes.",
            ],
        }
