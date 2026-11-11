"""
YOLOv8 Training Engine for AI-Based Kidney Stone Detection
Trains Ultralytics YOLOv8n on verified bounding-box annotated dataset.
Enforces CUDA GPU requirement, validates dataset before training,
supports training resumption, and saves best/last checkpoints to models/detection/.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import shutil
import sys
import json
import time
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DETECTION_MODELS_DIR,
    YOLO_DATA_YAML,
    YOLO_REPORTS_DIR,
    get_hardware_info,
)
from training.validate_yolo_dataset import validate_yolo_dataset, print_validation_report

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def check_cuda_requirement() -> Tuple[bool, str]:
    """
    Check CUDA GPU availability strictly.
    Does NOT silently fall back to CPU.
    """
    if not torch.cuda.is_available():
        return False, (
            "CRITICAL: CUDA GPU is not available in PyTorch environment!\n"
            "This project requires NVIDIA CUDA GPU execution (e.g., NVIDIA RTX 5060).\n"
            "Silent fallback to CPU is disallowed. Please verify your NVIDIA drivers and PyTorch CUDA build."
        )

    try:
        # Verify active tensor execution on CUDA device
        test_tensor = torch.zeros(1, device="cuda") + 1.0
        del test_tensor
    except Exception as e:
        return False, f"CRITICAL: CUDA execution test failed on GPU: {str(e)}"

    gpu_name = torch.cuda.get_device_name(0)
    vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1)
    return True, f"CUDA GPU active: '{gpu_name}' ({vram_mb} MB VRAM)"


def select_safe_batch_size(imgsz: int = 640) -> int:
    """
    Automatically select safe batch size based on available GPU VRAM.
    """
    if not torch.cuda.is_available():
        return 8

    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    if vram_mb >= 12000:
        return 32
    elif vram_mb >= 6000:
        return 16  # Safe for 8GB RTX 5060
    elif vram_mb >= 4000:
        return 8
    else:
        return 4


def train_yolo_detector(
    data_yaml: Optional[Path] = None,
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    batch_size: Optional[int] = None,
    img_size: int = 640,
    patience: int = 15,
    optimizer: str = "AdamW",
    seed: int = 42,
    resume: bool = False,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Train YOLOv8 on the validated kidney stone detection dataset.
    """
    if not HAS_ULTRALYTICS:
        raise ImportError("Ultralytics package is not installed. Run: pip install ultralytics")

    yaml_path = Path(data_yaml) if data_yaml else YOLO_DATA_YAML

    print("=" * 70)
    print("  YOLOv8 Detection Model Training — Kidney Stone AI")
    print("=" * 70)

    # 1. Pre-flight check: CUDA GPU requirement
    cuda_ok, cuda_msg = check_cuda_requirement()
    print(f"  [1/4] Hardware check: {cuda_msg}")
    if not cuda_ok:
        raise RuntimeError(cuda_msg)

    # 2. Pre-flight check: Dataset validation
    print(f"  [2/4] Validating YOLO dataset config: {yaml_path}...")
    val_report = validate_yolo_dataset(data_yaml_path=yaml_path, verify_image_files=False)
    if not val_report["valid"]:
        print_validation_report(val_report)
        err_msg = (
            "YOLO Training ABORTED: Dataset validation failed.\n"
            "Bounding-box annotations are missing or improperly formatted.\n"
            f"Details: {val_report['errors'][:3]}"
        )
        return {"success": False, "error": err_msg, "validation_report": val_report}

    print("  [✔] Dataset validation passed: Zero patient leakage & valid annotations.")

    # 3. Determine hyperparameters
    chosen_batch = batch_size if batch_size is not None else select_safe_batch_size(imgsz=img_size)
    print(f"  [3/4] Hyperparameters: Epochs={epochs}, Batch={chosen_batch}, ImgSz={img_size}, Optimizer={optimizer}, Patience={patience}")

    DETECTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    runs_dir = DETECTION_MODELS_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 4. Check for resumption if requested
    last_pt = DETECTION_MODELS_DIR / "last.pt"
    run_last_pt = runs_dir / "yolo_kidney_stone" / "weights" / "last.pt"

    if resume and (last_pt.exists() or run_last_pt.exists()):
        resume_checkpoint = str(last_pt if last_pt.exists() else run_last_pt)
        print(f"  Resuming training from checkpoint: {resume_checkpoint}")
        model = YOLO(resume_checkpoint)
        training_kwargs = {"resume": True}
    else:
        print(f"  Initializing model backbone: {model_name}...")
        model = YOLO(model_name)
        training_kwargs = {
            "data": str(yaml_path.resolve()),
            "epochs": epochs,
            "batch": chosen_batch,
            "imgsz": img_size,
            "patience": patience,
            "optimizer": optimizer,
            "seed": seed,
            "device": 0,  # CUDA GPU 0
            "workers": 0,  # Safe for Windows multiprocessing
            "project": str(runs_dir),
            "name": "yolo_kidney_stone",
            "exist_ok": True,
            "save": True,
            "save_period": 5,
            "plots": True,
            "verbose": True,
        }

    # 5. Start Training
    start_time = time.time()
    print("  [4/4] Executing Ultralytics YOLOv8 training loop on GPU...")

    try:
        results = model.train(**training_kwargs)
        duration_secs = round(time.time() - start_time, 1)

        # Standard checkpoints
        trained_weights_dir = runs_dir / "yolo_kidney_stone" / "weights"
        trained_best = trained_weights_dir / "best.pt"
        trained_last = trained_weights_dir / "last.pt"

        dest_best = DETECTION_MODELS_DIR / "best.pt"
        dest_last = DETECTION_MODELS_DIR / "last.pt"

        if trained_best.exists():
            shutil.copy2(trained_best, dest_best)
            print(f"\n  [✔] Best checkpoint saved to: {dest_best}")
        if trained_last.exists():
            shutil.copy2(trained_last, dest_last)
            print(f"  [✔] Last checkpoint saved to: {dest_last}")

        # Copy training curve plots into reports/yolo/
        run_folder = runs_dir / "yolo_kidney_stone"
        for plot_file in run_folder.glob("*.png"):
            shutil.copy2(plot_file, YOLO_REPORTS_DIR / plot_file.name)

        # Save training summary JSON
        summary = {
            "model": "yolov8n",
            "epochs_configured": epochs,
            "batch_size": chosen_batch,
            "img_size": img_size,
            "optimizer": optimizer,
            "patience": patience,
            "training_duration_seconds": duration_secs,
            "training_duration_minutes": round(duration_secs / 60.0, 2),
            "device": torch.cuda.get_device_name(0),
            "best_checkpoint": str(dest_best),
            "last_checkpoint": str(dest_last),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        history_path = DETECTION_MODELS_DIR / "yolo_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        reports_summary_path = YOLO_REPORTS_DIR / "training_summary.json"
        with open(reports_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print("  🎉 YOLOv8 Training Completed Successfully!")
        print(f"  Duration : {round(duration_secs/60.0, 1)} minutes")
        print(f"  Best Model Checkpoint : {dest_best}")
        print(f"  Reports & Plots       : {YOLO_REPORTS_DIR}")
        print("=" * 70)

        return {
            "success": True,
            "best_checkpoint": str(dest_best),
            "last_checkpoint": str(dest_last),
            "duration_seconds": duration_secs,
            "summary": summary,
        }

    except Exception as e:
        print(f"\n❌ YOLOv8 Training Error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "duration_seconds": round(time.time() - start_time, 1),
        }


if __name__ == "__main__":
    res = train_yolo_detector(epochs=100)
    if not res.get("success"):
        sys.exit(1)
