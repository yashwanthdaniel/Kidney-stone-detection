"""
Dataset Manager Page
Inspects dataset structure, validates classes, detects data leakage risks,
and partitions data into strictly patient-isolated train/val/test splits.
"""
import streamlit as st
from pathlib import Path
import json

from config.settings import (
    CLASS_NAMES,
    DEFAULT_RANDOM_SEED,
    SPLITS_DIR,
)
from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from training.inspect_dataset import (
    find_dataset_root,
    inspect_dataset_directory,
)
from training.split_dataset import (
    create_safe_dataset_split,
    load_split_manifest,
)

st.set_page_config(page_title="Dataset Manager | Kidney Stone AI", page_icon="📁", layout="wide")
inject_custom_css()

render_hero_banner(
    title="Dataset Management & Leakage-Free Splitting",
    subtitle="Configure dataset directories, inspect class distributions, and partition patient-isolated splits.",
    badge="DATA PIPELINE",
)

default_root = find_dataset_root()
dataset_path_input = st.text_input(
    "Dataset Root Directory",
    value=str(default_root) if default_root else "",
    help="Path containing Original and Augmented folders, or Stone/Non-Stone folders.",
)

col_actions = st.columns([1, 1, 2])
with col_actions[0]:
    inspect_btn = st.button("🔍 Inspect Dataset", use_container_width=True)
with col_actions[1]:
    verify_images = st.checkbox("Verify Image Integrity", value=False, help="Reads every image file to check for corrupt data (slower).")

if inspect_btn or "inspection_report" not in st.session_state:
    with st.spinner("Inspecting dataset directory structure and class balance..."):
        report = inspect_dataset_directory(dataset_path_input, verify_images=verify_images)
        st.session_state["inspection_report"] = report

report = st.session_state.get("inspection_report", {})

if report.get("found"):
    st.success(f"Discovered valid dataset at: `{report['dataset_root']}`")

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Images Discovered", f"{report['total_images']:,}")
    with m2:
        st.metric("Unique Patient IDs", f"{report['unique_patients_count']}")
    with m3:
        st.metric("Non-Stone Slices", f"{report['class_counts'].get('Non-Stone', 0):,}")
    with m4:
        st.metric("Stone Slices", f"{report['class_counts'].get('Stone', 0):,}")

    if report.get("leakage_warning"):
        st.warning(f"⚠️ {report['leakage_warning']}")

    # Directory Breakdown
    st.subheader("Subdirectory Breakdown")
    subsets_data = []
    for sub_name, counts in report.get("subset_counts", {}).items():
        subsets_data.append({
            "Directory Subset": sub_name,
            "Non-Stone": counts.get("Non-Stone", 0),
            "Stone": counts.get("Stone", 0),
            "Total Slices": counts.get("Non-Stone", 0) + counts.get("Stone", 0),
            "Unsupported Files": counts.get("unsupported", 0),
        })
    st.dataframe(subsets_data, use_container_width=True)

    if report.get("corrupt_files"):
        st.error(f"Found {len(report['corrupt_files'])} corrupt files!")
        st.json(report["corrupt_files"][:5])

    st.markdown("---")
    st.subheader("✂️ Dataset Splitting Configuration (Zero-Leakage)")
    st.caption(
        "Splitting isolates all slices and augmentations of each patient into a single partition. "
        "Test and validation sets evaluate exclusively on held-out original clinical slices."
    )

    s_c1, s_c2, s_c3, s_c4 = st.columns(4)
    with s_c1:
        train_ratio = st.slider("Train Ratio", min_value=0.50, max_value=0.85, value=0.70, step=0.05)
    with s_c2:
        val_ratio = st.slider("Validation Ratio", min_value=0.05, max_value=0.25, value=0.15, step=0.05)
    with s_c3:
        test_ratio = round(1.0 - train_ratio - val_ratio, 2)
        st.metric("Test Ratio (Calculated)", f"{test_ratio * 100:.0f}%")
    with s_c4:
        random_seed = st.number_input("Random Seed", value=DEFAULT_RANDOM_SEED, step=1)

    inc_aug = st.checkbox("Include Augmented Images in Val & Test", value=False, help="Recommended False: Evaluate validation and test sets strictly on real unaugmented clinical scans.")

    if st.button("🚀 Generate Safe Dataset Splits", type="primary", use_container_width=True):
        if test_ratio <= 0:
            st.error("Train ratio + Validation ratio must be less than 1.0 to preserve a test set.")
        else:
            with st.spinner("Generating patient-stratified partitions..."):
                try:
                    manifest = create_safe_dataset_split(
                        dataset_root=Path(report["dataset_root"]),
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                        random_seed=int(random_seed),
                        include_augmented_in_val_test=inc_aug,
                    )
                    st.success("Splits generated successfully with zero patient leakage!")
                    st.session_state["split_manifest"] = manifest
                except Exception as e:
                    st.error(f"Failed to create splits: {str(e)}")

    # Display Active Split Manifest
    manifest = st.session_state.get("split_manifest") or load_split_manifest()
    if manifest:
        st.subheader("📊 Active Split Manifest")
        sc = manifest.get("split_counts", {})
        split_display = []
        for s_name in ["train", "val", "test"]:
            if s_name in sc:
                item = sc[s_name]
                split_display.append({
                    "Partition": s_name.upper(),
                    "Total Images": item["total"],
                    "Non-Stone": item["by_class"].get("Non-Stone", 0),
                    "Stone": item["by_class"].get("Stone", 0),
                    "Original Slices": item["original_count"],
                    "Augmented Slices": item["augmented_count"],
                    "Patients Isolated": manifest.get("patient_counts", {}).get(s_name, 0),
                })
        st.dataframe(split_display, use_container_width=True)

        lv = manifest.get("leakage_verification", {})
        if lv.get("leakage_free"):
            st.success("Data Leakage Verification: Passed (0 overlapping parent images between train, val, and test).")
        else:
            st.error("Warning: Data leakage detected between partitions!")

else:
    st.error(f"Dataset root directory not found. Please verify the folder path above.")
    for err in report.get("errors", []):
        st.caption(f"Error detail: {err}")

render_disclaimer()
