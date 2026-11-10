"""
Weakly-Supervised YOLOv8 Dataset Builder & Detection Trainer
Extracts hyperdense radiopaque stone candidate bounding boxes from confirmed
stone CT slices, creates a standardized YOLO dataset, and trains YOLOv8.
Saves the resulting checkpoint to models/detection/best.pt.
"""
from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import yaml

from config.settings import SPLITS_DIR, DETECTION_MODELS_DIR, BASE_DIR

YOLO_DATASET_DIR = BASE_DIR / "data" / "yolo_dataset"


def extract_stone_bounding_boxes(image_path: str):
    """
    Detect hyperdense focal calcifications in CT scan slice.
    Returns list of [x_center, y_center, width, height] normalized.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape
    # Kidney stones typically have high Hounsfield radiodensity (brightest pixels)
    # Exclude outside borders/scanner markers by cropping center 85%
    mask_roi = np.zeros_like(img)
    margin_y = int(h * 0.08)
    margin_x = int(w * 0.08)
    mask_roi[margin_y:h - margin_y, margin_x:w - margin_x] = 255

    # Target top 0.5% - 2% intensity threshold
    p98 = np.percentile(img[mask_roi > 0], 98.5) if np.any(mask_roi > 0) else 200
    thresh_val = max(180, p98)

    _, binary = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, mask_roi)

    # Morphological opening to remove isolated noise pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter realistic kidney stone size (between 15 px^2 and 2500 px^2)
        if 15 <= area <= 2500:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Add slight padding around the detected calcification
            pad = 6
            bx1 = max(0, bx - pad)
            by1 = max(0, by - pad)
            bx2 = min(w, bx + bw + pad)
            by2 = min(h, by + bh + pad)

            # Convert to normalized YOLO coordinates: x_c, y_c, bw, bh
            xc = ((bx1 + bx2) / 2.0) / w
            yc = ((by1 + by2) / 2.0) / h
            norm_w = (bx2 - bx1) / w
            norm_h = (by2 - by1) / h

            boxes.append([xc, yc, norm_w, norm_h])

    return boxes


def prepare_yolo_dataset(max_train_samples: int = 500, max_val_samples: int = 100):
    """
    Build train/val YOLO directories and label files from split manifests.
    """
    print("[1/3] Preparing YOLOv8 dataset directory structure...")
    for split in ["train", "val"]:
        (YOLO_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"

    splits_info = [("train", train_csv, max_train_samples), ("val", val_csv, max_val_samples)]

    for split_name, csv_path, max_samples in splits_info:
        df = pd.read_csv(csv_path)
        # Sample both stone and non-stone images
        stone_df = df[df["class"] == "Stone"]
        nonstone_df = df[df["class"] == "Non-Stone"]

        n_stone = min(len(stone_df), max_samples // 2)
        n_ns = min(len(nonstone_df), max_samples // 2)

        sample_df = pd.concat([
            stone_df.sample(n=n_stone, random_state=42),
            nonstone_df.sample(n=n_ns, random_state=42),
        ]).reset_index(drop=True)

        print(f"  Processing {split_name} partition ({len(sample_df)} slices)...")

        for idx, row in sample_df.iterrows():
            img_src = Path(row["path"])
            if not img_src.exists():
                continue

            dest_img = YOLO_DATASET_DIR / "images" / split_name / f"{split_name}_{idx}_{img_src.name}"
            dest_lbl = YOLO_DATASET_DIR / "labels" / split_name / f"{dest_img.stem}.txt"

            shutil.copy(img_src, dest_img)

            if row["class"] == "Stone":
                boxes = extract_stone_bounding_boxes(str(img_src))
                with open(dest_lbl, "w") as f:
                    for b in boxes:
                        # class 0: Kidney Stone
                        f.write(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
            else:
                # Background negative sample
                dest_lbl.touch()

    # Create dataset YAML
    yaml_content = {
        "path": str(YOLO_DATASET_DIR.resolve()),
        "train": str((YOLO_DATASET_DIR / "images" / "train").resolve()),
        "val": str((YOLO_DATASET_DIR / "images" / "val").resolve()),
        "names": {0: "Kidney Stone"},
    }

    yaml_path = YOLO_DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"[2/3] Dataset YAML successfully created at: {yaml_path}")
    return yaml_path


def train_yolo(yaml_path: Path, epochs: int = 8, batch_size: int = 16):
    """
    Train YOLOv8n detector and copy best checkpoint to models/detection/best.pt.
    """
    print("[3/3] Training YOLOv8 object detector...")
    from ultralytics import YOLO

    DETECTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=384,
        project=str(DETECTION_MODELS_DIR / "runs"),
        name="kidney_stone_detector",
        exist_ok=True,
        verbose=True,
        device="cpu",
    )

    # Copy best.pt to standard path
    trained_best = DETECTION_MODELS_DIR / "runs" / "kidney_stone_detector" / "weights" / "best.pt"
    target_best = DETECTION_MODELS_DIR / "best.pt"

    if trained_best.exists():
        shutil.copy(trained_best, target_best)
        print(f"\n[SUCCESS] YOLOv8 trained and saved to: {target_best}")
        print(f"File size: {round(target_best.stat().st_size / (1024**2), 2)} MB")
        return target_best
    else:
        raise FileNotFoundError(f"Expected checkpoint not found at: {trained_best}")


if __name__ == "__main__":
    y_yaml = prepare_yolo_dataset(max_train_samples=400, max_val_samples=100)
    train_yolo(y_yaml, epochs=6, batch_size=16)
