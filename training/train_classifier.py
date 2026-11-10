"""
Classification Model Training Engine
Trains MobileNetV2, ResNet50, and EfficientNet-B0 with AdamW,
early stopping, validation monitoring, history saving, and progress callbacks.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.settings import (
    SPLITS_DIR,
    CLASSIFICATION_MODELS_DIR,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_IMAGE_SIZE,
    get_device,
)
from backend.services.preprocessing import KidneyStoneDataset, get_transforms
from backend.services.classification import (
    build_model,
    save_checkpoint,
    get_checkpoint_path,
    SUPPORTED_ARCHITECTURES,
)


def get_history_path(architecture: str) -> Path:
    """Path to the model's epoch training history JSON."""
    return CLASSIFICATION_MODELS_DIR / f"{architecture.lower()}_history.json"


def save_training_history(architecture: str, history: Dict[str, Any]) -> None:
    """Save epoch loss/accuracy logs to disk."""
    path = get_history_path(architecture)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def load_training_history(architecture: str) -> Optional[Dict[str, Any]]:
    """Load epoch loss/accuracy logs from disk."""
    path = get_history_path(architecture)
    if not path.exists():
        alt_path = CLASSIFICATION_MODELS_DIR / f"{architecture.lower().replace('_', '')}_history.json"
        if alt_path.exists():
            path = alt_path
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def train_single_classifier(
    architecture: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    patience: int = DEFAULT_PATIENCE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train a single classification architecture on the prepared dataset splits.
    """
    arch = architecture.lower()
    if arch not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{architecture}'. Supported: {SUPPORTED_ARCHITECTURES}")

    torch.manual_seed(random_seed)
    if device is None:
        device = get_device()

    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Dataset splits not found in {SPLITS_DIR}. Please generate dataset splits first."
        )

    # Prepare datasets and dataloaders
    train_dataset = KidneyStoneDataset(
        csv_file_or_df=train_csv,
        transform=get_transforms(split="train", apply_augmentation=True),
        max_samples=max_train_samples,
    )
    val_dataset = KidneyStoneDataset(
        csv_file_or_df=val_csv,
        transform=get_transforms(split="val", apply_augmentation=False),
        max_samples=max_val_samples,
    )

    # Windows PyTorch DataLoader num_workers=0 avoids multiprocessing spawn issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize model
    model = build_model(arch, num_classes=2, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    history: Dict[str, Any] = {
        "architecture": arch,
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "best_val_acc": 0.0,
        "stopped_early": False,
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "total_train_samples": len(train_dataset),
        "total_val_samples": len(val_dataset),
    }

    best_checkpoint_path = get_checkpoint_path(arch)
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    training_config = {
        "architecture": arch,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "patience": patience,
        "random_seed": random_seed,
        "device": str(device),
    }

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = running_train_loss / max(1, train_total)
        epoch_train_acc = (train_correct / max(1, train_total)) * 100.0

        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        epoch_val_loss = running_val_loss / max(1, val_total)
        epoch_val_acc = (val_correct / max(1, val_total)) * 100.0

        # Update History
        history["epochs"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 2))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 2))

        # Check for improvement
        improved = False
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            history["best_epoch"] = epoch
            history["best_val_loss"] = round(best_val_loss, 4)
            history["best_val_acc"] = round(best_val_acc, 2)
            patience_counter = 0
            improved = True

            # Save best checkpoint
            save_checkpoint(
                model=model,
                architecture=arch,
                path=best_checkpoint_path,
                epoch=epoch,
                best_val_acc=best_val_acc,
                best_val_loss=best_val_loss,
                training_config=training_config,
            )
        else:
            patience_counter += 1

        # Notify progress callback
        epoch_info = {
            "architecture": arch,
            "epoch": epoch,
            "total_epochs": epochs,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 2),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 2),
            "best_val_loss": round(best_val_loss, 4),
            "best_val_acc": round(best_val_acc, 2),
            "improved": improved,
            "patience_counter": patience_counter,
        }

        if progress_callback:
            progress_callback(epoch_info)

        # Early stopping check
        if patience_counter >= patience:
            history["stopped_early"] = True
            break

    total_duration = time.time() - start_training_time
    history["duration_seconds"] = round(total_duration, 1)

    # Save final history file
    save_training_history(arch, history)

    return {
        "architecture": arch,
        "success": True,
        "best_epoch": history["best_epoch"],
        "best_val_acc": history["best_val_acc"],
        "best_val_loss": history["best_val_loss"],
        "checkpoint_path": str(best_checkpoint_path),
        "history": history,
    }


def train_all_classifiers(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    patience: int = DEFAULT_PATIENCE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train MobileNetV2, ResNet50, and EfficientNet-B0 sequentially to conserve memory.
    """
    results = {}
    for arch in SUPPORTED_ARCHITECTURES:
        res = train_single_classifier(
            architecture=arch,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            random_seed=random_seed,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            progress_callback=progress_callback,
            device=device,
        )
        results[arch] = res

    return results
