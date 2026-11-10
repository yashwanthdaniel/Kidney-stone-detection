"""
Dataset Inspection and Validation Engine
Discovers dataset directories, handles casing/spelling variations,
detects corrupt images, maps patient IDs and parent-augmented relationships,
and generates validation reports to prevent data leakage.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import re
from PIL import Image
import json

from config.settings import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_NAMES,
    SUPPORTED_IMAGE_EXTS,
    BASE_DIR,
)


def find_dataset_root(candidate_path: Optional[str] = None) -> Optional[Path]:
    """
    Search for dataset root containing 'Original' and/or 'Augmented' folders.
    Tolerates nesting and spelling variations (e.g., 'Kindey Stone Dataset').
    """
    candidates = []
    if candidate_path:
        candidates.append(Path(candidate_path))

    # Standard workspace search paths
    candidates.extend([
        BASE_DIR / "dataset",
        BASE_DIR / "data" / "dataset",
        BASE_DIR,
    ])

    for base in candidates:
        if not base.exists():
            continue
        # Check direct or recursive subdirectories
        for p in [base] + list(base.rglob("*")):
            if not p.is_dir():
                continue
            subdirs = [d.name.lower() for d in p.iterdir() if d.is_dir()]
            has_orig = any("orig" in s for s in subdirs)
            has_aug = any("aug" in s for s in subdirs)
            if has_orig or has_aug:
                return p
    return None


def extract_patient_and_parent_id(filename: str) -> Tuple[Optional[str], str]:
    """
    Extract patient ID and base parent image ID from filename to prevent data leakage.
    Examples:
    'P001_FA_M_NS_I01.jpg' -> Patient: 'P001', Parent: 'P001_FA_M_NS_I01'
    'P001_FA_M_NS_I01_CR.jpg' -> Patient: 'P001', Parent: 'P001_FA_M_NS_I01'
    """
    stem = Path(filename).stem

    # Patient ID pattern (e.g., P001, P037, Patient_12, etc.)
    patient_match = re.match(r"^(P\d+|Patient[_\-\d]+)", stem, re.IGNORECASE)
    patient_id = patient_match.group(1).upper() if patient_match else None

    # Known augmentation suffixes in dataset
    # E.g. _CR, _DO, _FH, _FV, _GB, _EL, _ROT, _FLIP, _AUG, etc.
    parent_id = re.sub(r"(_CR|_DO|_FH|_FV|_GB|_EL|_ROT\d*|_FLIP\w*|_AUG\d*)$", "", stem, flags=re.IGNORECASE)

    return patient_id, parent_id


def inspect_dataset_directory(
    dataset_root: Optional[Union[str, Path]] = None,
    verify_images: bool = False
) -> Dict[str, Any]:
    """
    Inspect the dataset folder structure, collect statistics,
    and report any issues.
    """
    if dataset_root is None:
        root_path = find_dataset_root()
    else:
        root_path = Path(dataset_root)

    report: Dict[str, Any] = {
        "found": False,
        "dataset_root": str(root_path) if root_path else None,
        "directories": {},
        "classes_found": [],
        "missing_classes": [],
        "total_images": 0,
        "class_counts": {c: 0 for c in CLASS_NAMES},
        "subset_counts": {},
        "unique_patients_count": 0,
        "unique_parent_images_count": 0,
        "corrupt_files": [],
        "unsupported_files": [],
        "leakage_warning": None,
        "status": "NOT_FOUND",
        "errors": [],
    }

    if not root_path or not root_path.exists():
        report["errors"].append(f"Dataset root directory not found: {dataset_root or 'Default search paths'}")
        return report

    report["found"] = True

    # Find Original and Augmented subdirectories (case-insensitive)
    orig_dir = None
    aug_dir = None
    for d in root_path.iterdir():
        if d.is_dir():
            name_lower = d.name.lower()
            if "orig" in name_lower and orig_dir is None:
                orig_dir = d
            elif "aug" in name_lower and aug_dir is None:
                aug_dir = d

    # If nested deeper, search inside
    if not orig_dir and not aug_dir:
        for d in root_path.rglob("*"):
            if d.is_dir():
                name_lower = d.name.lower()
                if "orig" in name_lower and orig_dir is None:
                    orig_dir = d
                elif "aug" in name_lower and aug_dir is None:
                    aug_dir = d

    subsets = {}
    if orig_dir:
        subsets["Original"] = orig_dir
    if aug_dir:
        subsets["Augmented"] = aug_dir

    if not subsets:
        # Check if root itself contains Stone / Non-Stone
        subsets["Root"] = root_path

    report["directories"] = {k: str(v) for k, v in subsets.items()}

    all_patients: Set[str] = set()
    all_parent_keys: Set[str] = set()
    classes_discovered: Set[str] = set()
    image_records: List[Dict[str, Any]] = []

    for subset_name, subset_path in subsets.items():
        subset_stats: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
        subset_stats["unsupported"] = 0

        # Look for Stone and Non-Stone folders
        for class_dir in subset_path.iterdir():
            if not class_dir.is_dir():
                continue

            c_name = class_dir.name.strip()
            # Match canonical class names
            matched_class = None
            if re.search(r"non[_\-\s]?stone", c_name, re.IGNORECASE):
                matched_class = "Non-Stone"
            elif re.search(r"stone", c_name, re.IGNORECASE):
                matched_class = "Stone"

            if not matched_class:
                continue

            classes_discovered.add(matched_class)

            for f in class_dir.iterdir():
                if f.is_dir():
                    continue

                ext = f.suffix.lower()
                if ext not in SUPPORTED_IMAGE_EXTS:
                    report["unsupported_files"].append(str(f))
                    subset_stats["unsupported"] += 1
                    continue

                # Quick integrity check if requested
                if verify_images:
                    try:
                        with Image.open(f) as img:
                            img.verify()
                    except Exception as err:
                        report["corrupt_files"].append({"path": str(f), "error": str(err)})
                        continue

                pat_id, parent_id = extract_patient_and_parent_id(f.name)
                if pat_id:
                    all_patients.add(pat_id)
                parent_key = f"{matched_class}:{parent_id}"
                all_parent_keys.add(parent_key)

                report["class_counts"][matched_class] += 1
                subset_stats[matched_class] += 1
                report["total_images"] += 1

                image_records.append({
                    "path": str(f),
                    "filename": f.name,
                    "subset": subset_name,
                    "class": matched_class,
                    "class_idx": CLASS_TO_IDX[matched_class],
                    "patient_id": pat_id,
                    "parent_id": parent_id,
                    "parent_key": parent_key,
                })

        report["subset_counts"][subset_name] = subset_stats

    report["classes_found"] = sorted(list(classes_discovered))
    missing = [c for c in CLASS_NAMES if c not in classes_discovered]
    report["missing_classes"] = missing
    report["unique_patients_count"] = len(all_patients)
    report["unique_parent_images_count"] = len(all_parent_keys)

    # Assess leakage risk
    if "Original" in subsets and "Augmented" in subsets:
        report["leakage_warning"] = (
            "CRITICAL: Dataset contains both Original and Augmented subsets derived from the same base images. "
            "Random naive splitting WILL cause severe data leakage. "
            "Grouped splitting by parent image / patient ID MUST be used to isolate test/validation subjects."
        )

    if missing:
        report["status"] = "INCOMPLETE"
        report["errors"].append(f"Missing required class directories: {missing}")
    elif report["total_images"] == 0:
        report["status"] = "EMPTY"
        report["errors"].append("No valid images found in the inspected dataset directories.")
    else:
        report["status"] = "VALID"

    return report


if __name__ == "__main__":
    rep = inspect_dataset_directory(verify_images=False)
    print(json.dumps(rep, indent=2))
