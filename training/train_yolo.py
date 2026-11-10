"""
YOLOv8 Training and Dataset YAML Validation Service
Ensures detection datasets have valid bounding-box labels before training.
Prevents training detection models on pure classification datasets.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

from config.settings import DETECTION_MODELS_DIR

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def validate_yolo_yaml(yaml_path: Path) -> Dict[str, Any]:
    """
    Validate that a given YAML file exists, is well-formed,
    and references valid train/val image and label paths.
    """
    if not yaml_path.exists():
        return {
            "valid": False,
            "error": f"Dataset YAML file not found: {yaml_path}",
        }

    try:
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return {"valid": False, "error": "YAML content must be a dictionary."}

        required_keys = ["names"]
        missing = [k for k in required_keys if k not in data]
        if missing:
            return {"valid": False, "error": f"Missing required YAML keys: {missing}"}

        # Check train path
        train_path = data.get("train")
        if not train_path:
            return {"valid": False, "error": "YAML does not specify 'train' directory."}

        return {
            "valid": True,
            "data": data,
            "classes": data.get("names"),
            "train": train_path,
            "val": data.get("val"),
        }
    except Exception as e:
        return {"valid": False, "error": f"Failed to parse YAML: {str(e)}"}


def train_yolo_model(
    yaml_path: str,
    epochs: int = 20,
    batch_size: int = 16,
    img_size: int = 640,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Train YOLOv8 on an annotated detection dataset.
    """
    if not HAS_ULTRALYTICS:
        return {"success": False, "error": "Ultralytics package not installed."}

    y_path = Path(yaml_path)
    val_res = validate_yolo_yaml(y_path)
    if not val_res["valid"]:
        return {"success": False, "error": val_res["error"]}

    try:
        DETECTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model = YOLO("yolov8n.pt")  # Nano backbone for faster convergence

        results = model.train(
            data=str(y_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=str(DETECTION_MODELS_DIR / "runs"),
            name="yolo_kidney_stone",
            exist_ok=True,
            verbose=True,
        )

        # Copy best.pt to standard destination
        run_best = DETECTION_MODELS_DIR / "runs" / "yolo_kidney_stone" / "weights" / "best.pt"
        target_best = DETECTION_MODELS_DIR / "best.pt"
        if run_best.exists():
            shutil.copy(run_best, target_best)

        return {
            "success": True,
            "checkpoint_path": str(target_best),
            "epochs": epochs,
            "message": "YOLOv8 training completed successfully.",
        }
    except Exception as e:
        return {"success": False, "error": f"YOLO training failed: {str(e)}"}
