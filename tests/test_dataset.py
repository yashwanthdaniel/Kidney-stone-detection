"""
Unit tests for dataset inspection, class mapping, and data leakage prevention.
"""
import pytest
from pathlib import Path
import tempfile
from PIL import Image

from config.settings import CLASS_TO_IDX, IDX_TO_CLASS, CLASS_NAMES, NUM_CLASSES
from training.inspect_dataset import extract_patient_and_parent_id, inspect_dataset_directory
from training.split_dataset import create_safe_dataset_split


def test_canonical_class_mapping():
    """Verify central canonical mapping is strictly 0=Non-Stone, 1=Stone."""
    assert CLASS_TO_IDX["Non-Stone"] == 0
    assert CLASS_TO_IDX["Stone"] == 1
    assert IDX_TO_CLASS[0] == "Non-Stone"
    assert IDX_TO_CLASS[1] == "Stone"
    assert CLASS_NAMES == ["Non-Stone", "Stone"]
    assert NUM_CLASSES == 2


def test_patient_and_parent_id_extraction():
    """Verify regex correctly parses patient ID and base parent image ID."""
    # Standard original image
    pat, parent = extract_patient_and_parent_id("P001_FA_M_NS_I01.jpg")
    assert pat == "P001"
    assert parent == "P001_FA_M_NS_I01"

    # Augmented derived images
    pat_cr, parent_cr = extract_patient_and_parent_id("P001_FA_M_NS_I01_CR.jpg")
    assert pat_cr == "P001"
    assert parent_cr == "P001_FA_M_NS_I01"

    pat_gb, parent_gb = extract_patient_and_parent_id("P037_FA_M_S_I01_GB.jpg")
    assert pat_gb == "P037"
    assert parent_gb == "P037_FA_M_S_I01"


def test_synthetic_dataset_leakage_free_split():
    """Test dataset splitting on synthetic directory to verify zero patient/parent leakage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)

        # Create simulated Original & Augmented structure with 10 synthetic patients
        for subset in ["Original", "Augmented"]:
            for cls_name in ["Non-Stone", "Stone"]:
                (root / subset / cls_name).mkdir(parents=True, exist_ok=True)

        # Generate synthetic images
        dummy_img = Image.new("RGB", (32, 32), color="gray")

        # Patients P101-P105: Non-Stone; P106-P110: Stone
        for p_idx in range(101, 106):
            p_id = f"P{p_idx}"
            # Original
            orig_name = f"{p_id}_FA_M_NS_I01.jpg"
            dummy_img.save(root / "Original" / "Non-Stone" / orig_name)
            # Augmented
            dummy_img.save(root / "Augmented" / "Non-Stone" / f"{p_id}_FA_M_NS_I01_CR.jpg")
            dummy_img.save(root / "Augmented" / "Non-Stone" / f"{p_id}_FA_M_NS_I01_GB.jpg")

        for p_idx in range(106, 111):
            p_id = f"P{p_idx}"
            orig_name = f"{p_id}_FA_M_S_I01.jpg"
            dummy_img.save(root / "Original" / "Stone" / orig_name)
            dummy_img.save(root / "Augmented" / "Stone" / f"{p_id}_FA_M_S_I01_CR.jpg")
            dummy_img.save(root / "Augmented" / "Stone" / f"{p_id}_FA_M_S_I01_GB.jpg")

        out_splits = root / "splits"
        manifest = create_safe_dataset_split(
            dataset_root=root,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
            include_augmented_in_val_test=False,
            output_dir=out_splits,
        )

        assert manifest["leakage_verification"]["leakage_free"] is True
        assert manifest["leakage_verification"]["train_val_overlap"] == 0
        assert manifest["leakage_verification"]["train_test_overlap"] == 0
        assert manifest["leakage_verification"]["val_test_overlap"] == 0
        assert manifest["split_counts"]["train"]["total"] > 0
        assert manifest["split_counts"]["val"]["total"] > 0
        assert manifest["split_counts"]["test"]["total"] > 0
