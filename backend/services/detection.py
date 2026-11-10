"""
YOLOv8 Object Detection Service
Handles stone localization when valid bounding-box annotations and checkpoints are present.
Strictly separates classification from detection and prevents fake bounding boxes.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import numpy as np

from config.settings import DETECTION_MODELS_DIR

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def get_detection_checkpoint_path() -> Path:
    """Return default checkpoint path for YOLOv8 detector."""
    return DETECTION_MODELS_DIR / "best.pt"


def is_detector_available() -> bool:
    """Check if a valid trained YOLOv8 model checkpoint is available on disk."""
    cp = get_detection_checkpoint_path()
    return cp.exists() and cp.stat().st_size > 1024


def run_yolo_detection(
    image: Image.Image,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Run YOLOv8 object detection on an input CT image.
    If checkpoint is absent, returns an explicit unavailable message.
    """
    if not HAS_ULTRALYTICS:
        return {
            "available": False,
            "status": "Detection model not trained or unavailable. Ultralytics package required for YOLO.",
            "detections": [],
            "annotated_image": None,
        }

    checkpoint_path = get_detection_checkpoint_path()
    if not is_detector_available():
        return {
            "available": False,
            "status": "Detection model not trained or unavailable. Classification dataset does not contain bounding boxes.",
            "checkpoint_path": str(checkpoint_path),
            "detections": [],
            "annotated_image": None,
            "message": "To train YOLOv8, provide annotated bounding-box datasets (YAML format) in YOLO Detection tab.",
        }

    try:
        from config.settings import get_device
        device_str = str(get_device())
        model = YOLO(str(checkpoint_path))
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device_str,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        annotated_img_pil = None

        if results and len(results) > 0:
            res = results[0]
            # Convert plotted numpy array back to PIL
            plotted_bgr = res.plot()
            annotated_img_pil = Image.fromarray(plotted_bgr[..., ::-1])

            boxes = res.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, f"Class_{cls_id}")
                    detections.append({
                        "box_xyxy": [round(v, 2) for v in xyxy],
                        "width_px": round(xyxy[2] - xyxy[0], 2),
                        "height_px": round(xyxy[3] - xyxy[1], 2),
                        "confidence": round(conf, 4),
                        "class_id": cls_id,
                        "class_name": cls_name,
                    })

        return {
            "available": True,
            "status": "Success",
            "checkpoint_path": str(checkpoint_path),
            "num_detections": len(detections),
            "detections": detections,
            "annotated_image": annotated_img_pil,
            "conf_threshold": conf_threshold,
        }

    except Exception as e:
        return {
            "available": False,
            "status": f"Detection error: {str(e)}",
            "detections": [],
            "annotated_image": None,
        }
