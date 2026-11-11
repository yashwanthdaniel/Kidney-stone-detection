"""
YOLOv8 Dataset Preparation & Assembly Engine for Kidney Stone Detection
Prepares standardized YOLOv8 directory structure from patient-isolated splits,
copies images, pairs existing YOLO-format bounding box annotations,
creates empty label files for negative background slices (Non-Stone),
generates data.yaml, and strictly verifies patient-level isolation.

IMPORTANT: Does NOT fabricate bounding boxes or assume whole images are stones.
Requires genuine bounding-box annotations (LabelImg, CVAT, or YOLO format .txt).
"""
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import shutil
import sys
import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    BASE_DIR,
    DATA_DIR,
    SPLITS_DIR,
    YOLO_DATASET_DIR,
    YOLO_DATA_YAML,
    SUPPORTED_IMAGE_EXTS,
)
from training.split_dataset import create_safe_dataset_split, load_split_manifest
from training.validate_yolo_dataset import validate_yolo_dataset, print_validation_report


def discover_annotations_directory(candidate_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Search for directory containing existing YOLO .txt annotation files.
    """
    search_paths: List[Path] = []
    if candidate_dir:
        search_paths.append(Path(candidate_dir))

    search_paths.extend([
        BASE_DIR / "dataset" / "labels",
        BASE_DIR / "dataset" / "Original" / "labels",
        BASE_DIR / "dataset" / "annotations",
        DATA_DIR / "annotations",
        BASE_DIR / "annotations",
    ])

    for p in search_paths:
        if p.exists() and p.is_dir():
            txt_files = list(p.glob("*.txt"))
            if len(txt_files) > 0:
                return p
    return None


def prepare_yolo_dataset(
    annotation_source_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    force_rebuild: bool = True,
) -> Dict[str, Any]:
    """
    Assemble the complete YOLO dataset from classification splits and annotations:
    1. Ensure train/val/test splits exist with strict patient isolation.
    2. Build data/yolo_dataset/images/{train,val,test} and labels/{train,val,test}.
    3. Copy images.
    4. For Stone images, copy matching YOLO .txt label from annotation source.
    5. For Non-Stone images, generate empty .txt files (negative background samples).
    6. Generate data.yaml.
    7. Execute validation checks.
    """
    target_root = Path(output_dir) if output_dir else YOLO_DATASET_DIR
    anno_dir = discover_annotations_directory(annotation_source_dir)

    print("=" * 70)
    print("  YOLOv8 Dataset Preparation Engine — Kidney Stone Detection")
    print("=" * 70)
    print(f"  Target Directory      : {target_root}")
    print(f"  Annotation Source Dir : {anno_dir if anno_dir else 'None found (Checking dataset dirs)'}")

    # 1. Verify / generate classification splits
    manifest = load_split_manifest(SPLITS_DIR)
    if not manifest:
        print("  Generating patient-isolated splits (70% train / 15% val / 15% test)...")
        manifest = create_safe_dataset_split()

    splits = ["train", "val", "test"]

    # 2. Reset or ensure directories
    if force_rebuild and target_root.exists():
        for split in splits:
            img_s = target_root / "images" / split
            lbl_s = target_root / "labels" / split
            if img_s.exists():
                shutil.rmtree(img_s)
            if lbl_s.exists():
                shutil.rmtree(lbl_s)

    for split in splits:
        (target_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 3. Process each split according to CSV
    stats: Dict[str, Any] = {
        split: {
            "images_copied": 0,
            "stone_annotated": 0,
            "stone_missing_anno": 0,
            "non_stone_negative": 0,
        }
        for split in splits
    }

    missing_stone_annotations: List[str] = []
    annotated_stone_count = 0

    for split in splits:
        csv_file = SPLITS_DIR / f"{split}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Required split CSV not found: {csv_file}")

        df = pd.read_csv(csv_file)
        dest_img_dir = target_root / "images" / split
        dest_lbl_dir = target_root / "labels" / split

        for _, row in df.iterrows():
            src_img_path = Path(row["path"])
            if not src_img_path.exists():
                continue

            dest_img_path = dest_img_dir / src_img_path.name
            dest_lbl_path = dest_lbl_dir / f"{src_img_path.stem}.txt"

            # Copy image
            if not dest_img_path.exists():
                shutil.copy2(src_img_path, dest_img_path)
            stats[split]["images_copied"] += 1

            cls_name = row["class"]

            if cls_name == "Stone":
                # Look for matching label file
                found_label: Optional[Path] = None

                # Check annotation directory if discovered
                if anno_dir:
                    candidate = anno_dir / f"{src_img_path.stem}.txt"
                    if candidate.exists():
                        found_label = candidate

                # Check beside original image or in parent's labels folder
                if not found_label:
                    beside = src_img_path.parent / f"{src_img_path.stem}.txt"
                    sibling_labels = src_img_path.parent.parent / "labels" / f"{src_img_path.stem}.txt"
                    if beside.exists():
                        found_label = beside
                    elif sibling_labels.exists():
                        found_label = sibling_labels

                if found_label:
                    shutil.copy2(found_label, dest_lbl_path)
                    stats[split]["stone_annotated"] += 1
                    annotated_stone_count += 1
                else:
                    # Missing genuine annotation - do not invent bounding boxes!
                    stats[split]["stone_missing_anno"] += 1
                    missing_stone_annotations.append(src_img_path.name)
                    # Do not create fake label file

            else:
                # Non-Stone: Create valid empty negative label file
                with open(dest_lbl_path, "w", encoding="utf-8") as f:
                    pass  # 0 bytes, clean negative sample
                stats[split]["non_stone_negative"] += 1

    # 4. Generate data.yaml (compatible with Ultralytics YOLOv8)
    # Using relative paths from YOLO_DATASET_DIR or absolute paths
    yaml_dict = {
        "path": str(target_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "kidney_stone",
        },
    }

    yaml_file = target_root / "data.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"\n  [✔] data.yaml created at: {yaml_file}")

    # 5. Run validation
    validation_rep = validate_yolo_dataset(target_root, yaml_file, verify_image_files=False)

    print("-" * 70)
    print("  Preparation Summary:")
    for split in splits:
        s = stats[split]
        print(
            f"    • {split.upper():<5} : {s['images_copied']:>3} images | "
            f"Annotated Stones: {s['stone_annotated']:>3} | "
            f"Missing Stone Annos: {s['stone_missing_anno']:>3} | "
            f"Negative (Non-Stone): {s['non_stone_negative']:>3}"
        )

    print("-" * 70)
    if missing_stone_annotations:
        print(f"\n  ⚠️ NOTICE: {len(missing_stone_annotations)} stone slice(s) currently lack bounding-box annotations.")
        print("  In accordance with medical AI safety, bounding boxes are NEVER fabricated.")
        print("  Use LabelImg or CVAT to create YOLO format .txt annotations for stone images.")
        print("  Place them in 'dataset/labels/' and rerun preparation.")

    return {
        "success": validation_rep["valid"],
        "dataset_root": str(target_root),
        "data_yaml": str(yaml_file),
        "annotated_stone_count": annotated_stone_count,
        "missing_stone_annotations_count": len(missing_stone_annotations),
        "missing_examples": missing_stone_annotations[:10],
        "stats": stats,
        "validation_report": validation_rep,
    }


if __name__ == "__main__":
    result = prepare_yolo_dataset()
    print_validation_report(result["validation_report"])
