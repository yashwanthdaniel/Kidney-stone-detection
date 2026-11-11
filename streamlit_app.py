"""
AI-Based Kidney Stone Detection & Classification Research Application
Main entry point and multi-model operational dashboard.
Supports MobileNetV2, ResNet50, EfficientNet-B0, Ensemble, and YOLOv8 object detection.
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
from backend.database.db_manager import get_all_examinations
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
    subtitle="Integrated multi-model research platform evaluating MobileNetV2, ResNet50, EfficientNet-B0, Ensemble, and YOLOv8 on CT imaging.",
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
        <div class="metric-subtitle">{'Weights Loaded (best.pt)' if is_detector_available() else 'No Checkpoint'}</div>
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
st.subheader("🔬 CT Scan Analysis & Model Inference")

# Model Selection Option
col_sel, col_thresh = st.columns([2, 1])
with col_sel:
    selected_model = st.selectbox(
        "🎯 Select Model for Inference:",
        options=[
            "Overview (All Models)",
            "YOLOv8",
            "ResNet50",
            "MobileNetV2",
            "EfficientNet-B0",
            "Ensemble",
        ],
        index=0,
        help="Choose a specific neural network architecture or view multi-model consensus.",
    )

with col_thresh:
    if selected_model in ["YOLOv8", "Overview (All Models)"]:
        yolo_conf = st.slider("YOLO Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
    else:
        st.write("")  # Placeholder alignment

st.write("Upload a CT scan image, DICOM file, or radiology PDF. PDF text is extracted separately; only selected embedded images are sent to the image models.")

uploaded_file = st.file_uploader(
    "Upload CT Scan Image / DICOM / Radiology PDF",
    type=["png", "jpg", "jpeg", "dcm", "dicom", "pdf"],
    key="home_uploader",
)

analysis_images = []

if uploaded_file is not None:
    file_suffix = Path(uploaded_file.name).suffix.lower()

    if file_suffix == ".pdf":
        try:
            from backend.services.pdf_processor import extract_pdf_content
            pdf_result = extract_pdf_content(uploaded_file.getvalue())
        except ImportError:
            st.error("PDF support requires PyMuPDF. Install it with: python -m pip install pymupdf")
            st.stop()
        except Exception as e:
            st.error(f"Could not process PDF: {e}")
            st.stop()

        st.subheader("📄 Radiology PDF")
        st.success(f"PDF processed: {pdf_result['page_count']} page(s)")
        if pdf_result["text"].strip():
            with st.expander("View extracted radiology report text", expanded=False):
                st.text_area("Extracted PDF text", pdf_result["text"], height=220)
        else:
            st.info("No selectable text was found. This may be a scanned PDF.")

        pdf_images = pdf_result["images"]
        if not pdf_images:
            st.warning(
                "No embedded images were found in this PDF. The four image models need actual CT scan images. "
                "Upload the CT images separately; report text alone cannot be passed to YOLO or the classifiers."
            )
        else:
            st.write(f"Found {len(pdf_images)} embedded image(s). Select the actual CT image(s) to analyse.")
            image_labels = [f"Page {item['page']} — image {item['index']}" for item in pdf_images]
            selected_pdf_images = st.multiselect(
                "Select CT images extracted from the PDF",
                options=list(range(len(pdf_images))),
                default=list(range(len(pdf_images))),
                format_func=lambda i: image_labels[i],
                key="pdf_ct_image_selection",
            )
            for selected_index in selected_pdf_images:
                item = pdf_images[selected_index]
                analysis_images.append((
                    item["image"].convert("RGB"),
                    {"format": "PDF embedded image", "is_dicom": False, "pixel_spacing_mm": None},
                    f"{uploaded_file.name} — {image_labels[selected_index]}"
                ))
    else:
        try:
            image, meta = load_image(uploaded_file)
            analysis_images.append((image, meta, uploaded_file.name))
        except Exception as e:
            st.error(f"Failed to load image: {str(e)}")
            st.stop()

if analysis_images:
    for image_index, (image, meta, source_name) in enumerate(analysis_images, start=1):
        if len(analysis_images) > 1:
            st.markdown(f"### CT image {image_index} of {len(analysis_images)}")
        col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(image, caption=f"Scan: {source_name}", use_container_width=True)

        with col_info:
            st.markdown("### Image Specifications")
            st.write(f"- **Format:** `{meta.get('format') or ('DICOM' if meta.get('is_dicom') else '2D Image')}`")
            st.write(f"- **Dimensions:** `{image.width} x {image.height} pixels`")
            st.write(f"- **DICOM Calibrated:** `{'Yes' if meta.get('is_dicom') else 'No (Standard 2D pixels)'}`")
            if meta.get("pixel_spacing_mm"):
                st.write(f"- **Pixel Spacing:** `{meta['pixel_spacing_mm']} mm/pixel`")
            st.write(f"- **Active Model Selection:** `{selected_model}`")

        st.markdown("---")

        # =========================================================================
        # CASE 1: YOLOv8 Selected Specifically
        # =========================================================================
        if selected_model == "YOLOv8":
            st.subheader("🎯 YOLOv8 Object Detection (Localization)")

            if not is_detector_available():
                st.error("❌ **YOLOv8 Checkpoint Missing**")
                st.warning(
                    "No trained YOLOv8 model found at `models/detection/best.pt`.\n\n"
                    "To use YOLOv8 detection:\n"
                    "1. Annotate kidney stone bounding boxes using LabelImg or CVAT in YOLO format (`.txt`).\n"
                    "2. Place annotations in `dataset/labels/` or run `python training/prepare_and_train_yolo.py`.\n"
                    "3. Train using `python training/train_yolo.py`.\n"
                    "4. Check the **8_YOLO_Detection** page for the step-by-step annotation and training guide."
                )
            else:
                with st.spinner("Executing YOLOv8 stone localization on CUDA GPU..."):
                    det_res = run_yolo_detection(image, conf_threshold=yolo_conf)

                if det_res.get("available"):
                    c_ann, c_det = st.columns([1, 1])
                    with c_ann:
                        if det_res.get("annotated_image"):
                            st.image(
                                det_res["annotated_image"],
                                caption="YOLOv8 Bounding-Box Detection",
                                use_container_width=True,
                            )
                    with c_det:
                        num_stones = det_res.get("num_detections", 0)
                        if num_stones > 0:
                            st.success(f"🎯 **{num_stones} Kidney Stone(s) Detected**")
                        else:
                            st.info("ℹ️ **No kidney stones detected** above the confidence threshold.")

                        st.markdown(f"**Inference Latency:** `{det_res.get('inference_time_ms', 0)} ms`")
                        st.markdown(f"**Confidence Threshold:** `{det_res.get('conf_threshold')}`")
                        st.markdown(f"**Compute Device:** `{det_res.get('device', 'CUDA')}`")

                        if num_stones > 0:
                            st.markdown("#### Bounding Box Coordinates & Confidence")
                            for idx, det in enumerate(det_res.get("detections", []), 1):
                                with st.expander(f"Stone #{idx} — Confidence: {round(det['confidence']*100, 1)}%", expanded=True):
                                    st.write(f"- **Class:** `{det['class_name']}` (ID: {det['class_id']})")
                                    st.write(f"- **Confidence:** `{round(det['confidence']*100, 2)}%`")
                                    st.write(f"- **Pixel Box [x1, y1, x2, y2]:** `{det['box_xyxy_pixels']}`")
                                    st.write(f"- **Dimensions:** `{det['width_px']} x {det['height_px']} px`")
                                    st.write(f"- **Normalized YOLO [xc, yc, w, h]:** `{det['box_normalized_yolo']}`")
                else:
                    st.error(f"YOLOv8 execution failed: {det_res.get('status')}")

        # =========================================================================
        # CASE 2: Single Classifier Selected (ResNet50 / MobileNetV2 / EfficientNet)
        # =========================================================================
        elif selected_model in ["ResNet50", "MobileNetV2", "EfficientNet-B0"]:
            arch_map = {
                "ResNet50": "resnet50",
                "MobileNetV2": "mobilenet_v2",
                "EfficientNet-B0": "efficientnet_b0",
            }
            target_arch = arch_map[selected_model]
            st.subheader(f"🧠 {selected_model} Classifier Prediction")

            with st.spinner(f"Running {selected_model} inference..."):
                all_clf = run_all_available_classifiers(image)
                res = all_clf.get(target_arch, {})

            if res.get("available"):
                pred_cls = res["predicted_class"]
                conf_pct = round(res["confidence"] * 100, 1)
                stone_pct = round(res["stone_probability"] * 100, 1)
                non_stone_pct = round(res["non_stone_probability"] * 100, 1)
                badge_class = "model-badge-stone" if pred_cls == "Stone" else "model-badge-nonstone"

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown(f"""
                    <div class="model-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span style="font-weight: 700; font-size: 20px; color: #F8FAFC;">{selected_model}</span>
                            <span class="{badge_class}">{pred_cls.upper()}</span>
                        </div>
                        <div style="font-size: 32px; font-weight: 800; color: #38BDF8; margin-bottom: 8px;">{conf_pct}% <span style="font-size: 14px; color: #94A3B8; font-weight: 400;">confidence</span></div>
                        <div style="margin-bottom: 8px; font-size: 15px;"><span style="color: #EF4444; font-weight: 600;">Stone Probability:</span> {stone_pct}%</div>
                        <div style="margin-bottom: 12px; font-size: 15px;"><span style="color: #10B981; font-weight: 600;">Non-Stone Probability:</span> {non_stone_pct}%</div>
                        <div style="font-size: 12px; color: #64748B; border-top: 1px solid #334155; padding-top: 8px;">Inference Latency: {res.get('inference_time_ms', 0)} ms</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown("#### Probability Distribution")
                    st.progress(res["stone_probability"], text=f"Stone: {stone_pct}%")
                    st.progress(res["non_stone_probability"], text=f"Non-Stone: {non_stone_pct}%")
            else:
                st.warning(f"Model checkpoint for {selected_model} is not available. Train it in the Training Center first.")

        # =========================================================================
        # CASE 3: Ensemble Selected
        # =========================================================================
        elif selected_model == "Ensemble":
            st.subheader("⚖️ Multi-Model Ensemble Consensus")
            with st.spinner("Synthesizing multi-model ensemble..."):
                all_clf = run_all_available_classifiers(image)
                ens_res = compute_ensemble_prediction(all_clf)

            if ens_res.get("available"):
                ens_pred = ens_res["predicted_class"]
                ens_conf = round(ens_res["confidence"] * 100, 1)
                ens_badge = "model-badge-stone" if ens_pred == "Stone" else "model-badge-nonstone"

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="metric-title">Ensemble Consensus</span>
                            <span class="{ens_badge}">{ens_pred.upper()}</span>
                        </div>
                        <div class="metric-value">{ens_conf}% <span style="font-size: 14px; font-weight: 400; color: #94A3B8;">Confidence</span></div>
                        <div style="margin-top: 8px; font-size: 14px;">
                            <b>Stone:</b> {round(ens_res['stone_probability'] * 100, 1)}% | 
                            <b>Non-Stone:</b> {round(ens_res['non_stone_probability'] * 100, 1)}%
                        </div>
                        <div style="font-size: 12px; color: #64748B; margin-top: 6px;">
                            Contributing Classifiers: {", ".join([m.upper() for m in ens_res['contributing_models']])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**Model Disagreement:** `{'Flagged (Models Differ)' if ens_res.get('model_disagreement') else 'Full Consensus'}`")
                    st.markdown(f"**Consensus Score:** `{ens_res.get('consensus_percentage', 100)}%`")
                    st.json(ens_res.get("weights", {}))
            else:
                st.warning("Ensemble prediction unavailable. At least one trained classification model is required.")

        # =========================================================================
        # CASE 4: Overview (All Models)
        # =========================================================================
        else:
            st.subheader("🤖 Independent Deep Learning Classifiers")
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
                                No trained checkpoint found.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

            # Ensemble Result
            st.markdown("---")
            st.subheader("⚖️ Ensemble Consensus")
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
                st.warning("Ensemble prediction unavailable.")

            # YOLO Detector Status
            st.markdown("---")
            st.subheader("🎯 YOLOv8 Object Detection (Localization)")
            if is_detector_available():
                det_res = run_yolo_detection(image, conf_threshold=yolo_conf)
                if det_res.get("available"):
                    col_yimg, col_yinfo = st.columns([1, 1])
                    with col_yimg:
                        if det_res.get("annotated_image"):
                            st.image(det_res["annotated_image"], caption="YOLOv8 Stone Localization", use_container_width=True)
                    with col_yinfo:
                        st.success(f"Detected {det_res.get('num_detections', 0)} region(s) in {det_res.get('inference_time_ms', 0)} ms.")
                        st.json(det_res.get("detections", []))
            else:
                st.info("ℹ️ YOLOv8 detection model not trained. Checkpoint `models/detection/best.pt` not found. Stone bounding-box annotations required.")

else:
    st.caption("Upload a CT scan image, DICOM file, or PDF containing embedded CT images to run multi-model analysis.")

render_disclaimer()
