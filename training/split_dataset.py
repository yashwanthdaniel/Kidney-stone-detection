
"""
Dataset Splitting Engine
Patient-level and parent-image isolation to prevent data leakage.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import random
import json
import pandas as pd

from config.settings import (
    SPLITS_DIR,
    CLASS_TO_IDX,
    CLASS_NAMES,
    DEFAULT_RANDOM_SEED,
)
from training.inspect_dataset import (
    find_dataset_root,
    extract_patient_and_parent_id,
)


def create_safe_dataset_split(
    dataset_root: Optional[Path] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = DEFAULT_RANDOM_SEED,
    include_augmented_in_val_test: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-4:
        raise ValueError("Split ratios must sum to 1.0")

    if dataset_root is None:
        dataset_root = find_dataset_root()
        if not dataset_root:
            raise FileNotFoundError("Dataset root not found.")
    else:
        dataset_root = Path(dataset_root)

    out_dir = Path(output_dir) if output_dir else SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover images and identify their patient IDs
    all_images: List[Dict[str, Any]] = []

    for f in dataset_root.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in {
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
        }:
            continue

        parts = [p.lower() for p in f.parts]

        matched_class = None

        for p in parts:
            if "non-stone" in p or "non_stone" in p or "nonstone" in p:
                matched_class = "Non-Stone"
                break
            elif "stone" in p:
                matched_class = "Stone"

        if matched_class is None:
            continue

        is_aug = any("aug" in p for p in parts)
        subset_type = "Augmented" if is_aug else "Original"

        patient_id, parent_id = extract_patient_and_parent_id(f.name)

        # IMPORTANT:
        # Do not include the class in the group key.
        # The same patient must remain in one split across both classes.
        if patient_id:
            group_key = f"patient:{patient_id}"
        else:
            group_key = f"image:{parent_id}"

        all_images.append({
            "path": str(f.resolve()),
            "filename": f.name,
            "class": matched_class,
            "class_idx": CLASS_TO_IDX[matched_class],
            "subset": subset_type,
            "patient_id": patient_id or "UNKNOWN",
            "parent_id": parent_id,
            "group_key": group_key,
        })

    if not all_images:
        raise ValueError("No dataset images were found.")

    # 2. Group all images globally by patient or parent image.
    # This ensures Stone and Non-Stone images of one patient
    # cannot be assigned to different partitions.
    group_to_images: Dict[str, List[Dict[str, Any]]] = {}

    for item in all_images:
        group_key = item["group_key"]
        group_to_images.setdefault(group_key, []).append(item)

    all_groups = sorted(group_to_images.keys())

    if len(all_groups) < 3:
        raise ValueError("At least 3 independent patient/image groups are required.")

    # 3. Randomly divide groups into train, validation and test.
    rng = random.Random(random_seed)
    rng.shuffle(all_groups)

    n_total = len(all_groups)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    n_train = max(1, min(n_train, n_total - 2))
    n_val = max(1, min(n_val, n_total - n_train - 1))
    n_test = n_total - n_train - n_val

    split_groups: Dict[str, Set[str]] = {
        "train": set(all_groups[:n_train]),
        "val": set(all_groups[n_train:n_train + n_val]),
        "test": set(all_groups[n_train + n_val:]),
    }

    # 4. Assign images to partitions.
    splits: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for split_name, group_keys in split_groups.items():
        for group_key in group_keys:
            for item in group_to_images[group_key]:

                # Keep validation and test sets restricted to originals.
                if (
                    split_name in ("val", "test")
                    and not include_augmented_in_val_test
                    and item["subset"] == "Augmented"
                ):
                    continue

                splits[split_name].append(item)

    # 5. Verify patient-level isolation.
    def get_patients(items):
        return {
            item["patient_id"]
            for item in items
            if item["patient_id"] != "UNKNOWN"
        }

    train_patients = get_patients(splits["train"])
    val_patients = get_patients(splits["val"])
    test_patients = get_patients(splits["test"])

    patient_overlap = {
        "train_val": train_patients & val_patients,
        "train_test": train_patients & test_patients,
        "val_test": val_patients & test_patients,
    }

    # 6. Verify parent-image isolation.
    train_parents = {x["parent_id"] for x in splits["train"]}
    val_parents = {x["parent_id"] for x in splits["val"]}
    test_parents = {x["parent_id"] for x in splits["test"]}

    parent_overlap = {
        "train_val": train_parents & val_parents,
        "train_test": train_parents & test_parents,
        "val_test": val_parents & test_parents,
    }

    if any(patient_overlap.values()):
        raise RuntimeError(
            "PATIENT DATA LEAKAGE DETECTED: "
            f"{ {k: len(v) for k, v in patient_overlap.items()} }"
        )

    if any(parent_overlap.values()):
        raise RuntimeError(
            "PARENT IMAGE LEAKAGE DETECTED: "
            f"{ {k: len(v) for k, v in parent_overlap.items()} }"
        )

    # 7. Verify both classes exist in every split.
    for split_name, items in splits.items():
        found_classes = {item["class"] for item in items}
        missing = set(CLASS_NAMES) - found_classes

        if missing:
            raise RuntimeError(
                f"{split_name} split is missing classes: {sorted(missing)}. "
                "Review patient grouping and dataset distribution."
            )

    # 8. Save CSV files and manifest.
    manifest: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "random_seed": random_seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_groups": n_total,
        "include_augmented_in_val_test": include_augmented_in_val_test,
        "split_group_counts": {
            "train": len(split_groups["train"]),
            "val": len(split_groups["val"]),
            "test": len(split_groups["test"]),
        },
        "patient_counts": {
            "train": len(train_patients),
            "val": len(val_patients),
            "test": len(test_patients),
        },
        "leakage_verification": {
            "patient_train_val_overlap": len(patient_overlap["train_val"]),
            "patient_train_test_overlap": len(patient_overlap["train_test"]),
            "patient_val_test_overlap": len(patient_overlap["val_test"]),
            "parent_train_val_overlap": len(parent_overlap["train_val"]),
            "parent_train_test_overlap": len(parent_overlap["train_test"]),
            "parent_val_test_overlap": len(parent_overlap["val_test"]),
            "leakage_free": True,
        },
        "split_counts": {},
    }

    for split_name, items in splits.items():
        csv_path = out_dir / f"{split_name}.csv"
        df = pd.DataFrame(items)
        df.to_csv(csv_path, index=False)

        class_counts = {
            c: int((df["class"] == c).sum())
            for c in CLASS_NAMES
        }

        manifest["split_counts"][split_name] = {
            "total": len(df),
            "by_class": class_counts,
            "original_count": int((df["subset"] == "Original").sum()),
            "augmented_count": int((df["subset"] == "Augmented").sum()),
            "csv_path": str(csv_path),
        }

    manifest_path = out_dir / "split_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_split_manifest(splits_dir: Optional[Path] = None):
    s_dir = Path(splits_dir) if splits_dir else SPLITS_DIR
    manifest_file = s_dir / "split_manifest.json"

    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    return None


if __name__ == "__main__":
    result = create_safe_dataset_split()
    print(json.dumps(result, indent=2))
