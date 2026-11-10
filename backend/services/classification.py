"""
Classification Service: MobileNetV2, ResNet50, EfficientNet-B0
Manages model architecture construction, checkpoint saving/loading,
and independent inference pipelines.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import time
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

from config.settings import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_NAMES,
    NUM_CLASSES,
    CLASSIFICATION_MODELS_DIR,
    MODEL_FILENAMES,
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_device,
)
from backend.services.preprocessing import get_transforms

SUPPORTED_ARCHITECTURES = ["mobilenet_v2", "resnet50", "efficientnet_b0"]


def build_model(architecture: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Construct model with custom classification head for kidney stone detection.
    Includes offline and network fallback.
    """
    arch = architecture.lower()

    def get_model_safely(builder_fn, weights_obj):
        if pretrained:
            try:
                return builder_fn(weights=weights_obj)
            except Exception as err:
                print(f"[Notice] Could not download remote weights ({err}). Initializing {arch} architecture directly.")
                return builder_fn(weights=None)
        return builder_fn(weights=None)

    if arch == "mobilenet_v2":
        model = get_model_safely(models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    elif arch == "resnet50":
        model = get_model_safely(models.resnet50, models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif arch == "efficientnet_b0":
        model = get_model_safely(models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unsupported architecture: '{architecture}'. Choose from {SUPPORTED_ARCHITECTURES}")


def get_checkpoint_path(architecture: str) -> Path:
    """Get the standard checkpoint path for a given architecture."""
    filename = MODEL_FILENAMES.get(architecture.lower(), f"{architecture.lower()}_best.pth")
    return CLASSIFICATION_MODELS_DIR / filename


def save_checkpoint(
    model: nn.Module,
    architecture: str,
    path: Path,
    epoch: int,
    best_val_acc: float,
    best_val_loss: float,
    training_config: Dict[str, Any],
) -> None:
    """
    Save model weights with comprehensive metadata and canonical class mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "architecture": architecture.lower(),
        "state_dict": model.state_dict(),
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "input_size": DEFAULT_IMAGE_SIZE,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "epoch": int(epoch),
        "timestamp": datetime.now().isoformat(),
        "training_config": training_config,
    }
    torch.save(metadata, path)


def load_checkpoint(
    architecture: str,
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model and verify architecture and class mapping compatibility.
    """
    if device is None:
        device = get_device()

    path = checkpoint_path or get_checkpoint_path(architecture)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Load on CPU first to avoid CUDA mismatch
    checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Corrupted or invalid checkpoint file at {path}")

    saved_arch = checkpoint.get("architecture")
    if saved_arch and saved_arch.lower() != architecture.lower():
        raise ValueError(
            f"Architecture mismatch! Target: {architecture}, Checkpoint: {saved_arch}. "
            f"Refusing to load weights into mismatched architecture."
        )

    model = build_model(architecture, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def is_model_available(architecture: str) -> bool:
    """Check if a trained checkpoint exists for the given architecture."""
    path = get_checkpoint_path(architecture)
    return path.exists() and path.stat().st_size > 0


def run_single_model_inference(
    model: nn.Module,
    image: Image.Image,
    architecture: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Execute inference on an image and return class probabilities, predicted class,
    confidence score, and inference latency.
    """
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    # Preprocessing
    transform = get_transforms(split="inference")
    input_tensor = transform(image).unsqueeze(0).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # Canonical order: 0 = Non-Stone, 1 = Stone
    non_stone_prob = float(probabilities[0])
    stone_prob = float(probabilities[1])

    if stone_prob >= non_stone_prob:
        predicted_class = "Stone"
        confidence = stone_prob
    else:
        predicted_class = "Non-Stone"
        confidence = non_stone_prob

    return {
        "architecture": architecture,
        "available": True,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "stone_probability": stone_prob,
        "non_stone_probability": non_stone_prob,
        "probabilities": {
            "Non-Stone": non_stone_prob,
            "Stone": stone_prob,
        },
        "inference_time_ms": latency_ms,
        "status": "Success",
    }


def run_all_available_classifiers(
    image: Image.Image,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run inference through every supported classifier independently.
    If a model lacks a trained checkpoint, returns an explicit unavailable status.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for arch in SUPPORTED_ARCHITECTURES:
        if not is_model_available(arch):
            results[arch] = {
                "architecture": arch,
                "available": False,
                "status": f"No trained checkpoint found for {arch}. Train the model in Training Center first.",
            }
            continue

        try:
            model, _ = load_checkpoint(arch, device=device)
            res = run_single_model_inference(model, image, arch, device=device)
            results[arch] = res
        except Exception as e:
            results[arch] = {
                "architecture": arch,
                "available": False,
                "status": f"Inference error: {str(e)}",
            }

    return results
