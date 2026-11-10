"""
AI-Based Kidney Stone Detection & Classification Research Application
Main entry point and multi-model operational dashboard.
"""
import streamlit as st
from PIL import Image
from pathlib import Path
import json

from config.settings import (
    CLASS_NAMES,
    get_hardware_info,
)
from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from backend.services.image_loader import load_image
from backend.services.classification import (
    run_all_available_classifiers,
    is_model_available,
    SUPPORTED_ARCHITECTURES,
)
from backend.services.ensemble import compute_ensemble_prediction
from backend.services.detection import run_yolo_detection, is_detector_available
from backend.services.measurement import calculate_region_measurement
from backend.database.db_manager import get_all_examinations, save_examination
from backend.reporting.pdf_generator import generate_pdf_report
from training.inspect_dataset import inspect_dataset_directory
from training.split_dataset import load_split_manifest

st.set_page_config(
    page_title="AI Kidney Stone Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

# Sidebar Navigation & System Telemetry
st.sidebar.title("🩺 Navigation Portal")
st.sidebar.caption("AI-Assisted Kidney Stone Research")

hw = get_hardware_info()
st.sidebar.markdown(f"**Compute:** `{hw['device_selected'].upper()}`")
st.sidebar.markdown(f"**PyTorch:** `v{hw['torch_version']}`")
if hw.get("gpu_name"):
    st.sidebar.markdown(f"**GPU:** `{hw['gpu_name']}`")
if hw.get("cuda_compatibility_warning"):
    st.sidebar.caption(f"⚠️ {hw['cuda_compatibility_warning']}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🧭 Application Modules
- **1. Dashboard**
- **2. Dataset Manager**
- **3. Training Center**
- **4. CT Image Analysis**
- **5. Model Comparison**
- **6. Training History**
- **7. Explainable AI**
- **8. YOLO Detection**
- **9. Model Management**
- **10. Patient History**
- **11. Reports**
""")
st.sidebar.info("Use the native Streamlit page menu above or explore features directly below.")

# Hero Banner
render_hero_banner(
    title="AI-Based Kidney Stone Detection & Classification System",
    subtitle="Integrated multi-model research platform evaluating MobileNetV2, ResNet50, EfficientNet-B0, and YOLOv8 on CT imaging.",
    badge="RESEARCH APPLICATION",
)

# Top Status Indicators
m1, m2, m3, m4 = st.columns(4)
rep = inspect_dataset_directory(verify_images=False)
manifest = load_split_manifest()

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Dataset Status</div>
        <div class="metric-value">{f"{rep['total_images']:,}" if rep.get("found") else "Not Loaded"}</div>
        <div class="metric-subtitle">{f"{rep['unique_patients_count']} Patients" if rep.get("found") else "Configure Dataset"}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    trained_count = sum(1 for arch in SUPPORTED_ARCHITECTURES if is_model_available(arch))
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Classifiers Ready</div>
        <div class="metric-value">{trained_count} / {len(SUPPORTED_ARCHITECTURES)}</div>
        <div class="metric-subtitle">MobileNetV2, ResNet50, EffNet</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    yolo_status = "Available" if is_detector_available() else "Unavailable"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">YOLOv8 Detector</div>
        <div class="metric-value" style="font-size: 20px;">{yolo_status.upper()}</div>
        <div class="metric-subtitle">{'Weights Loaded' if is_detector_available() else 'No Bounding Boxes'}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    recent_exams = get_all_examinations(limit=10)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Saved Studies</div>
        <div class="metric-value">{len(recent_exams)}</div>
        <div class="metric-subtitle">SQLite Historical DB</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Interactive Quick Analysis
st.subheader("🔬 Quick CT Scan Analysis & Inference")
st.write("Upload a patient CT scan image (PNG, JPG, JPEG, or DICOM) to evaluate across all trained neural networks:")

uploaded_file = st.file_uploader(
    "Upload CT Scan Slice",
    type=["png", "jpg", "jpeg", "dcm", "dicom"],
    key="home_uploader",
)

if uploaded_file is not None:
    try:
        image, meta = load_image(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        st.stop()

    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Uploaded CT scan slice", use_container_width=True)

    with col_info:
        st.markdown("### Image Specifications")
        st.write(f"- **Format:** `{meta.get('format') or ('DICOM' if meta.get('is_dicom') else '2D Image')}`")
        st.write(f"- **Dimensions:** `{image.width} x {image.height} pixels`")
        st.write(f"- **DICOM Calibrated:** `{'Yes' if meta.get('is_dicom') else 'No (Standard 2D pixels)'}`")
        if meta.get("pixel_spacing_mm"):
            st.write(f"- **Pixel Spacing:** `{meta['pixel_spacing_mm']} mm/pixel`")

    st.markdown("---")
    st.subheader("🤖 Multi-Model Classification Results")

    with st.spinner("Executing inference across all independent deep learning classifiers..."):
        clf_results = run_all_available_classifiers(image)

    cards = st.columns(3)
    for i, arch in enumerate(SUPPORTED_ARCHITECTURES):
        with cards[i]:
            res = clf_results.get(arch, {})
            display_name = arch.upper().replace("_", " ")

            if res.get("available"):
                pred_cls = res["predicted_class"]
                conf_pct = round(res["confidence"] * 100, 1)
                stone_pct = round(res["stone_probability"] * 100, 1)
                non_stone_pct = round(res["non_stone_probability"] * 100, 1)
                badge_class = "model-badge-stone" if pred_cls == "Stone" else "model-badge-nonstone"

                st.markdown(f"""
                <div class="model-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-weight: 700; font-size: 16px; color: #F8FAFC;">{display_name}</span>
                        <span class="{badge_class}">{pred_cls.upper()}</span>
                    </div>
                    <div style="font-size: 26px; font-weight: 800; color: #38BDF8; margin-bottom: 8px;">{conf_pct}% <span style="font-size: 13px; color: #94A3B8; font-weight: 400;">confidence</span></div>
                    <div style="margin-bottom: 6px; font-size: 13px;"><span style="color: #EF4444; font-weight: 600;">Stone:</span> {stone_pct}%</div>
                    <div style="margin-bottom: 12px; font-size: 13px;"><span style="color: #10B981; font-weight: 600;">Non-Stone:</span> {non_stone_pct}%</div>
                    <div style="font-size: 11px; color: #64748B; border-top: 1px solid #334155; padding-top: 8px;">Latency: {res.get('inference_time_ms', 0)} ms</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="model-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-weight: 700; font-size: 16px; color: #F8FAFC;">{display_name}</span>
                        <span class="model-badge-unavailable">NOT TRAINED</span>
                    </div>
                    <p style="color: #94A3B8; font-size: 12px; margin-top: 10px;">
                        No trained checkpoint found. Train this model in the Training Center first.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # Ensemble Result
    st.markdown("---")
    st.subheader("⚖️ Ensemble Prediction")
    ens_res = compute_ensemble_prediction(clf_results)

    if ens_res.get("available"):
        ens_pred = ens_res["predicted_class"]
        ens_conf = round(ens_res["confidence"] * 100, 1)
        ens_badge = "model-badge-stone" if ens_pred == "Stone" else "model-badge-nonstone"

        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-title">Ensemble Decision</span>
                <span class="{ens_badge}">{ens_pred.upper()}</span>
            </div>
            <div class="metric-value">{ens_conf}% <span style="font-size: 14px; font-weight: 400; color: #94A3B8;">Confidence</span></div>
            <div style="margin-top: 8px; font-size: 13px;">
                <b>Stone:</b> {round(ens_res['stone_probability'] * 100, 1)}% | 
                <b>Non-Stone:</b> {round(ens_res['non_stone_probability'] * 100, 1)}%
            </div>
            <div style="font-size: 12px; color: #64748B; margin-top: 6px;">
                Contributing Models: {", ".join([m.upper() for m in ens_res['contributing_models']])} | Disagreement: {'Yes' if ens_res.get('model_disagreement') else 'No'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Ensemble prediction unavailable. At least one trained model is required.")

    # YOLO Detector Status
    st.markdown("---")
    st.subheader("🎯 YOLOv8 Object Detection")
    det_res = run_yolo_detection(image)
    if det_res.get("available"):
        st.success(f"Detected {det_res.get('num_detections', 0)} region(s).")
    else:
        st.info("ℹ️ Detection model not trained or unavailable. Stone localization bounding-box annotations required.")

else:
    st.caption("Upload a CT scan above to see live multi-model predictions and ensemble analysis.")

render_disclaimer()
