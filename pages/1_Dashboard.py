"""
Dashboard Page: System Overview & Status Monitor
Displays real dataset statistics, checkpoint statuses, hardware specs,
and latest examination summaries without fabricated data.
"""
import streamlit as st
from pathlib import Path
import json

from config.settings import (
    CLASS_NAMES,
    CLASSIFICATION_MODELS_DIR,
    DETECTION_MODELS_DIR,
    get_hardware_info,
)
from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from backend.services.classification import is_model_available, SUPPORTED_ARCHITECTURES
from backend.services.detection import is_detector_available
from backend.database.db_manager import get_all_examinations
from training.inspect_dataset import inspect_dataset_directory
from training.split_dataset import load_split_manifest

st.set_page_config(page_title="Dashboard | Kidney Stone AI", page_icon="📊", layout="wide")
inject_custom_css()

render_hero_banner(
    title="Kidney Stone AI Research Dashboard",
    subtitle="Operational status of deep learning classifiers, YOLOv8 detector, dataset partitions, and hardware compute.",
    badge="SYSTEM STATUS",
)

# 1. Hardware & Runtime Specs
hw = get_hardware_info()
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Compute Device</div>
        <div class="metric-value">{hw['device_selected'].upper()}</div>
        <div class="metric-subtitle">PyTorch v{hw['torch_version']}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    gpu_label = hw['gpu_name'] or "Not Detected"
    if len(gpu_label) > 18:
        gpu_label = gpu_label[:16] + "..."
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">GPU Accelerator</div>
        <div class="metric-value" style="font-size: 18px;">{gpu_label}</div>
        <div class="metric-subtitle">VRAM: {hw['gpu_memory_mb'] or 0} MB</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Storage Available</div>
        <div class="metric-value">{hw['disk_free_gb'] or 'N/A'} GB</div>
        <div class="metric-subtitle">Total: {hw['disk_total_gb'] or 'N/A'} GB</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    exams = get_all_examinations(limit=1)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Saved Examinations</div>
        <div class="metric-value">{len(get_all_examinations(limit=500))}</div>
        <div class="metric-subtitle">SQLite Database Active</div>
    </div>
    """, unsafe_allow_html=True)

if hw.get("cuda_compatibility_warning"):
    st.warning(f"Hardware notice: {hw['cuda_compatibility_warning']}")

st.markdown("<br>", unsafe_allow_html=True)

# 2. Dataset & Splits Inspection
col_data, col_models = st.columns([1, 1])

with col_data:
    st.subheader("📁 Dataset Inspection & Splitting Status")
    manifest = load_split_manifest()
    rep = inspect_dataset_directory(verify_images=False)

    if rep.get("found"):
        st.success(f"Dataset Verified: {rep['total_images']:,} images discovered across {rep['unique_patients_count']} patients.")
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            st.metric("Non-Stone Slices", f"{rep['class_counts'].get('Non-Stone', 0):,}")
        with sub_c2:
            st.metric("Stone Slices", f"{rep['class_counts'].get('Stone', 0):,}")

        if manifest:
            st.info(
                f"**Splits Active (Patient-Isolated):** "
                f"Train: {manifest['split_counts']['train']['total']:,} | "
                f"Val: {manifest['split_counts']['val']['total']:,} | "
                f"Test: {manifest['split_counts']['test']['total']:,}"
            )
            if manifest["leakage_verification"].get("leakage_free"):
                st.caption("Zero data leakage verified across train/val/test patient partitions.")
        else:
            st.warning("No train/val/test split generated yet. Visit **Dataset Manager** to create partitions.")
    else:
        st.error(f"Dataset root directory not found. Please configure path in Dataset Manager.")

with col_models:
    st.subheader("🧠 Model Availability & Checkpoints")
    for arch in SUPPORTED_ARCHITECTURES:
        available = is_model_available(arch)
        badge = '<span class="model-badge-nonstone">TRAINED & READY</span>' if available else '<span class="model-badge-unavailable">NOT TRAINED</span>'
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: rgba(30,41,59,0.5); border: 1px solid #334155; border-radius: 8px; margin-bottom: 8px;">
            <span style="font-weight: 600; font-size: 14px;">{arch.upper().replace('_', ' ')}</span>
            {badge}
        </div>
        """, unsafe_allow_html=True)

    # YOLO Detector Status
    yolo_ready = is_detector_available()
    yolo_badge = '<span class="model-badge-nonstone">TRAINED</span>' if yolo_ready else '<span class="model-badge-unavailable">NO BOUNDING BOXES</span>'
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: rgba(30,41,59,0.5); border: 1px solid #334155; border-radius: 8px; margin-bottom: 8px;">
        <span style="font-weight: 600; font-size: 14px;">YOLOV8 OBJECT DETECTOR</span>
        {yolo_badge}
    </div>
    """, unsafe_allow_html=True)

# 3. Latest Examination
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📋 Latest Examination Record")
if exams:
    latest = exams[0]
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 16px; font-weight: 700; color: #38BDF8;">Exam #{latest['id']}</span>
            <span style="color: #94A3B8; font-size: 12px;">{latest['timestamp']}</span>
        </div>
        <p style="margin-top: 8px; margin-bottom: 4px; font-size: 14px;">
            <b>Predicted Class:</b> {latest.get('predicted_class') or 'N/A'} | 
            <b>Confidence:</b> {round((latest.get('confidence') or 0.0) * 100, 1)}% | 
            <b>Calibrated:</b> {'Yes' if latest.get('is_dicom') else 'No (2D Pixels)'}
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("No research examinations recorded yet. Run an analysis on the **CT Image Analysis** page to see results.")

render_disclaimer()
