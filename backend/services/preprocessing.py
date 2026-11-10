"""
Preprocessing and Dataset Pipelines for Kidney Stone Detection
Implements standardized transforms, training augmentations,
and PyTorch Dataset abstractions.
"""
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from config.settings import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CLASS_TO_IDX,
)
from backend.services.image_loader import load_image


def get_transforms(
    split: str = "train",
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    apply_augmentation: bool = True
) -> T.Compose:
    """
    Construct torchvision transform pipeline for training, validation, or inference.
    """
    if split == "train" and apply_augmentation:
        return T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class KidneyStoneDataset(Dataset):
    """
    PyTorch Dataset loading images from manifest CSV with patient/parent metadata.
    """
    def __init__(
        self,
        csv_file_or_df: Any,
        transform: Optional[T.Compose] = None,
        max_samples: Optional[int] = None,
    ):
        if isinstance(csv_file_or_df, (str, Path)):
            self.df = pd.read_csv(csv_file_or_df)
        elif isinstance(csv_file_or_df, pd.DataFrame):
            self.df = csv_file_or_df.copy()
        else:
            raise ValueError("Expected CSV path or pandas DataFrame")

        if max_samples and max_samples < len(self.df):
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        self.transform = transform or get_transforms(split="val")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_path = row["path"]
        class_name = row["class"]
        class_idx = int(row.get("class_idx", CLASS_TO_IDX[class_name]))

        try:
            image, meta = load_image(img_path)
        except Exception:
            # Fallback for rare unreadable files
            image = Image.new("RGB", (224, 224), color=0)
            meta = {"error": "Failed to read image"}

        tensor = self.transform(image)
        item_meta = {
            "path": img_path,
            "filename": row.get("filename", Path(img_path).name),
            "class": class_name,
            "patient_id": row.get("patient_id", "UNKNOWN"),
            "parent_id": row.get("parent_id", ""),
            "subset": row.get("subset", ""),
        }
        return tensor, class_idx, item_meta
