"""
YOLOv8 Independent Test Set Evaluation Engine for Kidney Stone Detection
Evaluates the trained YOLOv8 model (models/detection/best.pt) on the independent
test partition (data/yolo_dataset/images/test).
Computes Precision, Recall, mAP@50, mAP@50-95, inference latency, confusion matrix,
and exports publication-ready visual artifacts into reports/yolo/.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import shutil
import sys
import json
import time
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DETECTION_MODELS_DIR,
    YOLO_DATA_YAML,
    YOLO_REPORTS_DIR,
    YOLO_DATASET_DIR,
)

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def evaluate_yolo_on_test_set(
    checkpoint_path: Optional[Path] = None,
    data_yaml: Optional[Path] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    save_sample_images: int = 10,
) -> Dict[str, Any]:
    """
    Run evaluation on the test partition without inventing results.
    """
    if not HAS_ULTRALYTICS:
        raise ImportError("Ultralytics package is not installed. Run: pip install ultralytics")

    best_ckpt = Path(checkpoint_path) if checkpoint_path else (DETECTION_MODELS_DIR / "best.pt")
    yaml_path = Path(data_yaml) if data_yaml else YOLO_DATA_YAML

    if not best_ckpt.exists():
        raise FileNotFoundError(
            f"Trained YOLO checkpoint not found: {best_ckpt}\n"
            "Please train the model first before running evaluation."
        )

    if not yaml_path.exists():
        raise FileNotFoundError(f"YOLO dataset config not found: {yaml_path}")

    print("=" * 70)
    print("  YOLOv8 Independent Test Set Evaluation — Kidney Stone AI")
    print("=" * 70)
    print(f"  Checkpoint  : {best_ckpt}")
    print(f"  Dataset YAML: {yaml_path}")
    print(f"  Device      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    device_arg = 0 if torch.cuda.is_available() else "cpu"
    eval_output_dir = YOLO_REPORTS_DIR
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model and run test set validation
    print("\n  [1/3] Running Ultralytics validation on independent TEST split...")
    model = YOLO(str(best_ckpt))

    metrics = model.val(
        data=str(yaml_path.resolve()),
        split="test",
        conf=conf_threshold,
        iou=iou_threshold,
        device=device_arg,
        project=str(eval_output_dir / "val_runs"),
        name="test_evaluation",
        exist_ok=True,
        save_json=True,
        plots=True,
        verbose=True,
    )

    # 2. Extract metrics safely
    box_metrics = metrics.box
    precision = float(box_metrics.p[0]) if len(box_metrics.p) > 0 else float(box_metrics.mp)
    recall = float(box_metrics.r[0]) if len(box_metrics.r) > 0 else float(box_metrics.mr)
    map50 = float(box_metrics.map50)
    map50_95 = float(box_metrics.map)

    speed = metrics.speed
    preprocess_ms = round(speed.get("preprocess", 0.0), 2)
    inference_ms = round(speed.get("inference", 0.0), 2)
    loss_ms = round(speed.get("loss", 0.0), 2)
    postprocess_ms = round(speed.get("postprocess", 0.0), 2)
    total_latency_ms = round(preprocess_ms + inference_ms + postprocess_ms, 2)

    # 3. Generate sample detection overlays
    print(f"\n  [2/3] Generating sample detection overlays on test images (saving up to {save_sample_images})...")
    test_img_dir = YOLO_DATASET_DIR / "images" / "test"
    detections_dir = eval_output_dir / "test_detections"
    detections_dir.mkdir(parents=True, exist_ok=True)

    sample_results = []
    if test_img_dir.exists():
        test_images = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))
        for idx, img_path in enumerate(test_images[:save_sample_images]):
            pred = model.predict(
                source=str(img_path),
                conf=conf_threshold,
                iou=iou_threshold,
                device=device_arg,
                verbose=False,
            )
            if pred and len(pred) > 0:
                p_res = pred[0]
                save_path = detections_dir / f"test_pred_{idx:03d}_{img_path.name}"
                p_res.save(filename=str(save_path))
                sample_results.append({
                    "image": img_path.name,
                    "stones_detected": len(p_res.boxes),
                    "confidences": [round(float(c), 4) for c in p_res.boxes.conf.tolist()] if p_res.boxes is not None else [],
                    "saved_path": str(save_path),
                })

    # 4. Copy evaluation plots into reports/yolo/
    print("\n  [3/3] Archiving evaluation graphs and metrics...")
    val_run_folder = eval_output_dir / "val_runs" / "test_evaluation"
    if val_run_folder.exists():
        for f in val_run_folder.glob("*.png"):
            dest_f = eval_output_dir / f.name
            shutil.copy2(f, dest_f)

    # Load training history if available to include training duration
    history_file = DETECTION_MODELS_DIR / "yolo_history.json"
    training_duration_mins = None
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                h = json.load(f)
                training_duration_mins = h.get("training_duration_minutes")
        except Exception:
            pass

    report = {
        "model": "YOLOv8n",
        "checkpoint": str(best_ckpt),
        "split_evaluated": "test",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round((2 * precision * recall) / max(precision + recall, 1e-6), 4),
        "mAP_50": round(map50, 4),
        "mAP_50_95": round(map50_95, 4),
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "inference_latency_ms": {
            "preprocess": preprocess_ms,
            "inference": inference_ms,
            "postprocess": postprocess_ms,
            "total_latency": total_latency_ms,
        },
        "training_duration_minutes": training_duration_mins,
        "sample_detections_count": len(sample_results),
        "report_directory": str(eval_output_dir),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    report_path = eval_output_dir / "test_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("  📊 INDEPENDENT TEST SET EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Precision    : {report['precision'] * 100:.2f}%")
    print(f"  Recall       : {report['recall'] * 100:.2f}%")
    print(f"  F1-Score     : {report['f1_score'] * 100:.2f}%")
    print(f"  mAP@50       : {report['mAP_50'] * 100:.2f}%")
    print(f"  mAP@50-95    : {report['mAP_50_95'] * 100:.2f}%")
    print(f"  Inference    : {total_latency_ms} ms/slice (Model core: {inference_ms} ms)")
    if training_duration_mins:
        print(f"  Train Time   : {training_duration_mins} min")
    print(f"  Full Report  : {report_path}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    evaluate_yolo_on_test_set()

