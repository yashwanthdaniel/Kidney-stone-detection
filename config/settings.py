"""
Kidney Stone Detection & Classification System Configuration
"""
from pathlib import Path
import os
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

# Centralized Class Mapping (Canonical order: 0 = Non-Stone, 1 = Stone)
CLASS_TO_IDX = {"Non-Stone": 0, "Stone": 1}
IDX_TO_CLASS = {0: "Non-Stone", 1: "Stone"}
CLASS_NAMES = ["Non-Stone", "Stone"]
NUM_CLASSES = 2

# Directory Paths
DATA_DIR = BASE_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
CLASSIFICATION_MODELS_DIR = MODELS_DIR / "classification"
DETECTION_MODELS_DIR = MODELS_DIR / "detection"
DATABASE_DIR = BASE_DIR / "backend" / "database"
DATABASE_PATH = DATABASE_DIR / "research_history.db"
REPORTS_DIR = BASE_DIR / "reports"

# Model filenames
MODEL_FILENAMES = {
    "mobilenet_v2": "mobilenetv2_best.pth",
    "resnet50": "resnet50_best.pth",
    "efficientnet_b0": "efficientnet_b0_best.pth",
    "yolov8": "best.pt",
}

# Image Preprocessing & Normalization Defaults
DEFAULT_IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Supported formats
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_DICOM_EXTS = {".dcm", ".dicom"}

# Training Hyperparameter Defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 15
DEFAULT_PATIENCE = 4
DEFAULT_RANDOM_SEED = 42

# Ensure core directories exist
for directory in [
    DATA_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    CLASSIFICATION_MODELS_DIR,
    DETECTION_MODELS_DIR,
    DATABASE_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

def get_device() -> torch.device:
    """Return available torch device, verifying kernel execution compatibility."""
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device="cuda") + 1
            del test_tensor
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def get_hardware_info() -> dict:
    """Inspect and return system and hardware capabilities."""
    import platform
    import shutil
    import sys

    device = get_device()
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_selected": str(device),
        "gpu_name": None,
        "gpu_memory_mb": None,
        "disk_free_gb": None,
        "disk_total_gb": None,
        "cuda_compatibility_warning": None,
    }

    try:
        total, used, free = shutil.disk_usage(BASE_DIR)
        info["disk_free_gb"] = round(free / (2**30), 2)
        info["disk_total_gb"] = round(total / (2**30), 2)
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            mem_bytes = torch.cuda.get_device_properties(0).total_memory
            info["gpu_memory_mb"] = round(mem_bytes / (1024**2), 1)
        except Exception:
            pass

        if device.type == "cpu":
            info["cuda_compatibility_warning"] = (
                f"GPU '{info.get('gpu_name') or 'NVIDIA'}' was detected, but its compute architecture "
                "is not supported by the current CUDA runtime kernels. CPU fallback is active."
            )

    return info
