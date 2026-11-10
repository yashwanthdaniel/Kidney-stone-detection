"""
Dataset Splitting Engine with Patient-Level & Group-Level Isolation
Prevents data leakage by strictly partitioning patients and parent images
so no derived/augmented images cross partition boundaries.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import random
import json
import csv
import pandas as pd

from config.settings import (
    SPLITS_DIR,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_NAMES,
    DEFAULT_RANDOM_SEED,
)
from training.inspect_dataset import (
    find_dataset_root,
    extract_patient_and_parent_id,
    inspect_dataset_directory,
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
    """
    Partition dataset into train, validation, and test sets with strictly guaranteed
    patient and parent-image isolation.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-4:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    if dataset_root is None:
        dataset_root = find_dataset_root()
        if not dataset_root:
            raise FileNotFoundError("Dataset root directory not found.")
    else:
        dataset_root = Path(dataset_root)

    out_dir = Path(output_dir) if output_dir else SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover all images
    all_images: List[Dict[str, Any]] = []
    for f in dataset_root.rglob("*"):
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
            parts = [p.lower() for p in f.parts]
            # Identify class
            matched_class = None
            for p in parts:
                if "non-stone" in p or "non_stone" in p or "nonstone" in p:
                    matched_class = "Non-Stone"
                    break
                elif "stone" in p:
                    matched_class = "Stone"
            if not matched_class:
                continue

            is_aug = any("aug" in p for p in parts)
            subset_type = "Augmented" if is_aug else "Original"

            pat_id, parent_id = extract_patient_and_parent_id(f.name)
            # Group key prioritizes patient ID if available, else parent_id
            group_key = f"{matched_class}:{pat_id}" if pat_id else f"{matched_class}:{parent_id}"

            all_images.append({
                "path": str(f.resolve()),
                "filename": f.name,
                "class": matched_class,
                "class_idx": CLASS_TO_IDX[matched_class],
                "subset": subset_type,
                "patient_id": pat_id or "UNKNOWN",
                "parent_id": parent_id,
                "group_key": group_key,
            })

    if not all_images:
        raise ValueError("No images discovered to partition.")

    # 2. Group images by unique group_key (patient / root image) per class
    groups_by_class: Dict[str, List[str]] = {c: [] for c in CLASS_NAMES}
    group_to_images: Dict[str, List[Dict[str, Any]]] = {}

    for item in all_images:
        gk = item["group_key"]
        c = item["class"]
        if gk not in group_to_images:
            group_to_images[gk] = []
            groups_by_class[c].append(gk)
        group_to_images[gk].append(item)

    # 3. Stratified group splitting per class using random_seed
    rng = random.Random(random_seed)

    split_groups: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    for c in CLASS_NAMES:
        class_groups = sorted(groups_by_class[c])
        rng.shuffle(class_groups)
        n_total = len(class_groups)

        if n_total < 3:
            raise ValueError(f"Too few distinct subject groups for class '{c}' ({n_total} groups). Cannot create 3-way split.")

        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        # Ensure at least 1 group in val and test if possible
        n_train = max(1, min(n_train, n_total - 2))
        n_val = max(1, min(n_val, n_total - n_train - 1))
        n_test = n_total - n_train - n_val

        train_keys = class_groups[:n_train]
        val_keys = class_groups[n_train:n_train + n_val]
        test_keys = class_groups[n_train + n_val:]

        split_groups["train"].update(train_keys)
        split_groups["val"].update(val_keys)
        split_groups["test"].update(test_keys)

    # 4. Assign images to partitions
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for split_name in ["train", "val", "test"]:
        for gk in split_groups[split_name]:
            items = group_to_images[gk]
            for it in items:
                # If evaluating val/test strictly on clinical originals (recommended standard)
                if split_name in ["val", "test"] and not include_augmented_in_val_test:
                    if it["subset"] == "Augmented":
                        continue
                splits[split_name].append(it)

    # 5. Strict Data Leakage Verification
    train_patients = {x["patient_id"] for x in splits["train"] if x["patient_id"] != "UNKNOWN"}
    val_patients = {x["patient_id"] for x in splits["val"] if x["patient_id"] != "UNKNOWN"}
    test_patients = {x["patient_id"] for x in splits["test"] if x["patient_id"] != "UNKNOWN"}

    train_parents = {x["parent_id"] for x in splits["train"]}
    val_parents = {x["parent_id"] for x in splits["val"]}
    test_parents = {x["parent_id"] for x in splits["test"]}

    overlap_train_val = train_parents.intersection(val_parents)
    overlap_train_test = train_parents.intersection(test_parents)
    overlap_val_test = val_parents.intersection(test_parents)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED! Overlapping parent images: "
            f"Train-Val: {len(overlap_train_val)}, Train-Test: {len(overlap_train_test)}, Val-Test: {len(overlap_val_test)}"
        )

    # 6. Save split files
    manifest: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "random_seed": random_seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "include_augmented_in_val_test": include_augmented_in_val_test,
        "split_counts": {},
        "patient_counts": {
            "train": len(train_patients),
            "val": len(val_patients),
            "test": len(test_patients),
        },
        "leakage_verification": {
            "train_val_overlap": len(overlap_train_val),
            "train_test_overlap": len(overlap_train_test),
            "val_test_overlap": len(overlap_val_test),
            "leakage_free": True,
        },
    }

    for s_name, s_items in splits.items():
        csv_path = out_dir / f"{s_name}.csv"
        df = pd.DataFrame(s_items)
        df.to_csv(csv_path, index=False)

        class_counts = {c: int((df["class"] == c).sum()) if not df.empty else 0 for c in CLASS_NAMES}
        manifest["split_counts"][s_name] = {
            "total": len(df),
            "by_class": class_counts,
            "original_count": int((df["subset"] == "Original").sum()) if not df.empty else 0,
            "augmented_count": int((df["subset"] == "Augmented").sum()) if not df.empty else 0,
            "csv_path": str(csv_path),
        }

    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_split_manifest(splits_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load existing split manifest if available."""
    s_dir = Path(splits_dir) if splits_dir else SPLITS_DIR
    m_file = s_dir / "split_manifest.json"
    if m_file.exists():
        try:
            with open(m_file, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


if __name__ == "__main__":
    manifest = create_safe_dataset_split()
    print(json.dumps(manifest, indent=2))

