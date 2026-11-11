"""
YOLOv8 Dataset Validation Engine for Kidney Stone Detection
Validates YAML configuration, directory layout, image-label pairing,
bounding-box coordinate geometry, negative background images,
and strictly enforces patient-level partition isolation to prevent data leakage.
"""
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import sys
import yaml
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    YOLO_DATASET_DIR,
    YOLO_DATA_YAML,
    SUPPORTED_IMAGE_EXTS,
)
from training.inspect_dataset import extract_patient_and_parent_id


def validate_single_label_file(
    label_path: Path,
    expected_class_id: int = 0,
) -> Tuple[bool, List[str], int]:
    """
    Validate a single YOLO label file.
    Format per line: <class_id> <x_center> <y_center> <width> <height>
    Empty file is valid for negative background images (Non-Stone).
    Returns (is_valid, errors, box_count).
    """
    errors: List[str] = []
    box_count = 0

    if not label_path.exists():
        return False, [f"Label file missing: {label_path.name}"], 0

    try:
        content = label_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return False, [f"Could not read {label_path.name}: {str(e)}"], 0

    if not content:
        # Valid empty file for negative sample
        return True, [], 0

    lines = content.splitlines()
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(
                f"{label_path.name}:{line_num}: Expected 5 values (class x_c y_c w h), got {len(parts)} -> '{line}'"
            )
            continue

        # Parse class ID
        try:
            cls_id = int(parts[0])
            if cls_id != expected_class_id:
                errors.append(
                    f"{label_path.name}:{line_num}: Invalid class_id {cls_id} (expected {expected_class_id} for kidney_stone)"
                )
        except ValueError:
            errors.append(f"{label_path.name}:{line_num}: Non-integer class_id '{parts[0]}'")
            continue

        # Parse coordinates
        try:
            xc, yc, w, h = map(float, parts[1:5])
        except ValueError:
            errors.append(f"{label_path.name}:{line_num}: Non-numeric bounding box coordinates '{parts[1:5]}'")
            continue

        # Verify normalized range [0.0, 1.0]
        tol = 0.05  # Slight tolerance for edge annotations
        if not (-tol <= xc <= 1.0 + tol and -tol <= yc <= 1.0 + tol):
            errors.append(
                f"{label_path.name}:{line_num}: Center coords ({xc:.4f}, {yc:.4f}) out of [0, 1] range"
            )

        if w <= 0.0 or h <= 0.0:
            errors.append(
                f"{label_path.name}:{line_num}: Non-positive box dimensions: width={w:.4f}, height={h:.4f}"
            )

        if w > 1.0 + tol or h > 1.0 + tol:
            errors.append(
                f"{label_path.name}:{line_num}: Box dimensions exceed image bounds: width={w:.4f}, height={h:.4f}"
            )

        box_count += 1

    return len(errors) == 0, errors, box_count


def validate_yolo_dataset(
    dataset_dir: Optional[Path] = None,
    data_yaml_path: Optional[Path] = None,
    verify_image_files: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive validation of YOLO dataset:
    - YAML structure and paths
    - Train / Val / Test directory pairing
    - Coordinate ranges and label formatting
    - Negative sample presence
    - Patient-level split isolation
    """
    root = Path(dataset_dir) if dataset_dir else YOLO_DATASET_DIR
    yaml_file = Path(data_yaml_path) if data_yaml_path else (root / "data.yaml")

    report: Dict[str, Any] = {
        "valid": False,
        "yaml_path": str(yaml_file),
        "dataset_root": str(root),
        "errors": [],
        "warnings": [],
        "split_stats": {},
        "patient_isolation": {
            "verified": False,
            "train_patients": 0,
            "val_patients": 0,
            "test_patients": 0,
            "overlap_train_val": 0,
            "overlap_train_test": 0,
            "overlap_val_test": 0,
        },
        "total_images": 0,
        "total_boxes": 0,
        "positive_images": 0,
        "negative_images": 0,
    }

    # 1. Validate YAML existence and schema
    if not yaml_file.exists():
        report["errors"].append(f"YOLO dataset config YAML not found: {yaml_file}")
        return report

    try:
        with open(yaml_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
    except Exception as e:
        report["errors"].append(f"Failed to parse {yaml_file.name}: {str(e)}")
        return report

    if not isinstance(yaml_data, dict):
        report["errors"].append("YAML content must be a dictionary mapping.")
        return report

    # Check names
    names = yaml_data.get("names")
    if names is None:
        report["errors"].append("YAML missing required 'names' key.")
    elif isinstance(names, dict):
        if 0 not in names and "0" not in names:
            report["errors"].append("YAML 'names' dictionary must define class 0 (e.g., 0: kidney_stone).")
    elif isinstance(names, list):
        if len(names) == 0:
            report["errors"].append("YAML 'names' list is empty.")
    else:
        report["errors"].append("YAML 'names' must be a list or dictionary.")

    # 2. Check splits layout
    splits = ["train", "val", "test"]
    split_patients: Dict[str, Set[str]] = {s: set() for s in splits}
    total_boxes_overall = 0
    total_pos_images = 0
    total_neg_images = 0

    for split in splits:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.exists():
            report["errors"].append(f"Missing images directory for split '{split}': {img_dir}")
            continue

        if not lbl_dir.exists():
            report["errors"].append(f"Missing labels directory for split '{split}': {lbl_dir}")
            continue

        # Collect images
        image_files = [
            f for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]

        split_boxes = 0
        split_pos = 0
        split_neg = 0
        split_errors = 0

        for img_path in image_files:
            # Check image integrity if requested
            if verify_image_files:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as e:
                    report["errors"].append(f"Corrupted image file {img_path.name}: {str(e)}")
                    split_errors += 1
                    continue

            # Check matching label file
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                report["errors"].append(
                    f"Missing label file for image: {img_path.name} -> Expected: {lbl_path.name}"
                )
                split_errors += 1
                continue

            # Validate label content
            is_valid, lbl_errs, box_count = validate_single_label_file(lbl_path, expected_class_id=0)
            if not is_valid:
                report["errors"].extend(lbl_errs[:5])  # Cap at 5 errors per file to avoid flooding
                split_errors += len(lbl_errs)
                continue

            split_boxes += box_count
            if box_count > 0:
                split_pos += 1
            else:
                split_neg += 1

            # Extract patient ID for leakage checking
            pat_id, _ = extract_patient_and_parent_id(img_path.name)
            if pat_id and pat_id != "UNKNOWN":
                split_patients[split].add(pat_id)

        report["split_stats"][split] = {
            "image_count": len(image_files),
            "positive_images": split_pos,
            "negative_images": split_neg,
            "stone_boxes": split_boxes,
            "unique_patients": len(split_patients[split]),
        }

        total_boxes_overall += split_boxes
        total_pos_images += split_pos
        total_neg_images += split_neg
        report["total_images"] += len(image_files)

    report["total_boxes"] = total_boxes_overall
    report["positive_images"] = total_pos_images
    report["negative_images"] = total_neg_images

    # 3. Patient Isolation Verification
    train_pats = split_patients["train"]
    val_pats = split_patients["val"]
    test_pats = split_patients["test"]

    ov_tv = train_pats & val_pats
    ov_tt = train_pats & test_pats
    ov_vt = val_pats & test_pats

    report["patient_isolation"] = {
        "verified": len(ov_tv) == 0 and len(ov_tt) == 0 and len(ov_vt) == 0,
        "train_patients": len(train_pats),
        "val_patients": len(val_pats),
        "test_patients": len(test_pats),
        "overlap_train_val": len(ov_tv),
        "overlap_train_test": len(ov_tt),
        "overlap_val_test": len(ov_vt),
    }

    if ov_tv:
        report["errors"].append(
            f"DATA LEAKAGE: {len(ov_tv)} patient(s) shared between train and val splits: {sorted(list(ov_tv))[:5]}"
        )
    if ov_tt:
        report["errors"].append(
            f"DATA LEAKAGE: {len(ov_tt)} patient(s) shared between train and test splits: {sorted(list(ov_tt))[:5]}"
        )
    if ov_vt:
        report["errors"].append(
            f"DATA LEAKAGE: {len(ov_vt)} patient(s) shared between val and test splits: {sorted(list(ov_vt))[:5]}"
        )

    # 4. Overall sanity checks
    if report["total_images"] == 0:
        report["errors"].append("YOLO dataset is empty (0 images found across splits).")
    elif total_boxes_overall == 0:
        report["errors"].append(
            "No bounding-box annotations found! All label files are empty. YOLO cannot train without stone annotations."
        )

    if total_neg_images == 0 and total_pos_images > 0:
        report["warnings"].append(
            "Dataset does not contain negative background images (Non-Stone slices). Including negative images is strongly recommended to reduce false positives."
        )

    report["valid"] = len(report["errors"]) == 0
    return report


def print_validation_report(report: Dict[str, Any]) -> None:
    """Print readable validation summary to stdout."""
    print("=" * 70)
    print("  YOLOv8 Dataset Validation Report — Kidney Stone Detection")
    print("=" * 70)
    print(f"  Dataset Root : {report['dataset_root']}")
    print(f"  YAML Config  : {report['yaml_path']}")
    print(f"  Status       : {'PASSED ✅' if report['valid'] else 'FAILED ❌'}")
    print("-" * 70)

    print("  Split Breakdown:")
    for split, stats in report.get("split_stats", {}).items():
        print(
            f"    • {split.upper():<5} : {stats['image_count']:>4} images | "
            f"Stone: {stats['positive_images']:>3} | "
            f"Non-Stone: {stats['negative_images']:>3} | "
            f"Boxes: {stats['stone_boxes']:>4} | "
            f"Patients: {stats['unique_patients']:>3}"
        )

    iso = report["patient_isolation"]
    iso_status = "STRICT ISOLATION VERIFIED (Zero Leakage) ✅" if iso["verified"] else "DATA LEAKAGE DETECTED ❌"
    print(f"\n  Patient Isolation: {iso_status}")
    print(f"    Train patients: {iso['train_patients']} | Val patients: {iso['val_patients']} | Test patients: {iso['test_patients']}")
    if not iso["verified"]:
        print(f"    Overlaps: Train-Val={iso['overlap_train_val']}, Train-Test={iso['overlap_train_test']}, Val-Test={iso['overlap_val_test']}")

    if report["warnings"]:
        print("\n  ⚠️ Warnings:")
        for w in report["warnings"]:
            print(f"    - {w}")

    if report["errors"]:
        print("\n  ❌ Errors:")
        for e in report["errors"][:15]:
            print(f"    - {e}")
        if len(report["errors"]) > 15:
            print(f"    ... and {len(report['errors']) - 15} more error(s)")

    print("=" * 70)


if __name__ == "__main__":
    rep = validate_yolo_dataset()
    print_validation_report(rep)
    sys.exit(0 if rep["valid"] else 1)

