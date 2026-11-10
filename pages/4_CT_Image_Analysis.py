"""
CT Image Analysis Page
Uploads CT slices (PNG, JPG, JPEG, DICOM), executes individual model inference
(MobileNetV2, ResNet50, EfficientNet-B0), computes weighted ensemble prediction,
runs YOLOv8 detection if trained, and records study in database.
"""
import streamlit as st
from PIL import Image
import uuid

from config.settings import CLASS_NAMES
from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from backend.services.image_loader import load_image
from backend.services.classification import run_all_available_classifiers, SUPPORTED_ARCHITECTURES
from backend.services.ensemble import compute_ensemble_prediction
from backend.services.detection import run_yolo_detection
from backend.services.measurement import calculate_region_measurement
from backend.database.db_manager import save_examination
from backend.reporting.pdf_generator import generate_pdf_report

st.set_page_config(page_title="CT Image Analysis | Kidney Stone AI", page_icon="🔬", layout="wide")
inject_custom_css()

render_hero_banner(
    title="CT Image Analysis & Multi-Model Inference",
    subtitle="Evaluate patient CT scans across independent MobileNetV2, ResNet50, and EfficientNet-B0 classifiers with ensemble synthesis.",
    badge="INFERENCE WORKFLOW",
)

uploaded_file = st.file_uploader(
    "Upload Patient CT Slice (PNG, JPG, JPEG, DICOM)",
    type=["png", "jpg", "jpeg", "dcm", "dicom"],
    help="DICOM files preserve original physical millimeter pixel spacing; standard images are displayed in 2D pixels.",
)

if uploaded_file is not None:
    # 1. Load Image & Extract Physical Calibration
    try:
        image, meta = load_image(uploaded_file)
    except Exception as e:
        st.error(f"Failed to process uploaded file: {str(e)}")
        st.stop()

    img_col, meta_col = st.columns([1, 1])

    with img_col:
        st.subheader("Uploaded CT Scan Slice")
        st.image(image, caption=f"Scan: {uploaded_file.name}", use_container_width=True)

    with meta_col:
        st.subheader("Clinical Image Metadata")
        st.write(f"- **Format:** `{meta.get('format') or ('DICOM' if meta.get('is_dicom') else '2D Image')}`")
        st.write(f"- **Dimensions:** `{image.width} x {image.height} pixels`")
        st.write(f"- **DICOM Calibrated:** `{'Yes' if meta.get('is_dicom') else 'No (Standard 2D pixels)'}`")
        if meta.get("pixel_spacing_mm"):
            st.write(f"- **Pixel Spacing:** `{meta['pixel_spacing_mm']} mm/pixel`")
        if meta.get("slice_thickness_mm"):
            st.write(f"- **Slice Thickness:** `{meta['slice_thickness_mm']} mm`")

    st.markdown("---")

    # 2. Run Independent Deep Learning Classifiers
    st.subheader("🤖 Independent Deep Learning Model Predictions")
    st.caption("Each model evaluates the image independently. Class probabilities and inference times are strictly computed from loaded weights.")

    with st.spinner("Running inference across trained neural network architectures..."):
        clf_results = run_all_available_classifiers(image)

    cards = st.columns(3)
    arch_order = ["mobilenet_v2", "resnet50", "efficientnet_b0"]

    for i, arch in enumerate(arch_order):
        with cards[i]:
            res = clf_results.get(arch, {})
            model_display_name = arch.upper().replace("_", " ")

            if res.get("available"):
                pred_cls = res["predicted_class"]
                conf_pct = round(res["confidence"] * 100, 1)
                stone_pct = round(res["stone_probability"] * 100, 1)
                non_stone_pct = round(res["non_stone_probability"] * 100, 1)

                badge_class = "model-badge-stone" if pred_cls == "Stone" else "model-badge-nonstone"

                st.markdown(f"""
                <div class="model-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-weight: 700; font-size: 16px; color: #F8FAFC;">{model_display_name}</span>
                        <span class="{badge_class}">{pred_cls.upper()}</span>
                    </div>
                    <div style="font-size: 26px; font-weight: 800; color: #38BDF8; margin-bottom: 8px;">{conf_pct}% <span style="font-size: 13px; color: #94A3B8; font-weight: 400;">confidence</span></div>
                    <div style="margin-bottom: 6px; font-size: 13px;">
                        <span style="color: #EF4444; font-weight: 600;">Stone:</span> {stone_pct}%
                    </div>
                    <div style="margin-bottom: 12px; font-size: 13px;">
                        <span style="color: #10B981; font-weight: 600;">Non-Stone:</span> {non_stone_pct}%
                    </div>
                    <div style="font-size: 11px; color: #64748B; border-top: 1px solid #334155; padding-top: 8px;">
                        Latency: {res.get('inference_time_ms', 0)} ms
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="model-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-weight: 700; font-size: 16px; color: #F8FAFC;">{model_display_name}</span>
                        <span class="model-badge-unavailable">UNAVAILABLE</span>
                    </div>
                    <p style="color: #94A3B8; font-size: 12px; margin-top: 10px;">
                        No trained checkpoint available. Train this model in the Training Center first.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Ensemble Synthesis Section
    st.subheader("⚖️ Multi-Model Ensemble Synthesis")
    st.caption("Combines class probabilities of all available trained models using normalized weighted averaging.")

    ens_res = compute_ensemble_prediction(clf_results)

    if ens_res.get("available"):
        e1, e2 = st.columns([1, 1])
        with e1:
            ens_cls = ens_res["predicted_class"]
            ens_conf = round(ens_res["confidence"] * 100, 1)
            ens_badge = "model-badge-stone" if ens_cls == "Stone" else "model-badge-nonstone"

            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="metric-title">Ensemble Consensus</span>
                    <span class="{ens_badge}">{ens_cls.upper()}</span>
                </div>
                <div class="metric-value">{ens_conf}% <span style="font-size: 14px; font-weight: 400; color: #94A3B8;">Confidence</span></div>
                <div style="margin-top: 8px; font-size: 13px;">
                    <b>Stone:</b> {round(ens_res['stone_probability'] * 100, 1)}% | 
                    <b>Non-Stone:</b> {round(ens_res['non_stone_probability'] * 100, 1)}%
                </div>
                <div style="font-size: 12px; color: #64748B; margin-top: 6px;">
                    Contributing Classifiers: {", ".join([m.upper() for m in ens_res['contributing_models']])}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with e2:
            st.markdown(f"**Model Disagreement:** `{'Disagreement Flagged' if ens_res.get('model_disagreement') else 'Full Agreement (Consensus)'}`")
            st.markdown(f"**Consensus Percentage:** `{ens_res.get('consensus_percentage', 100)}%`")
            st.markdown(f"**Weighting Scheme:** `{ens_res.get('weights_type')}`")
            st.json(ens_res.get("weights", {}))
    else:
        st.warning("Ensemble prediction unavailable. At least one trained model is required.")

    st.markdown("---")

    # 4. YOLO Object Detection Section
    st.subheader("🎯 YOLOv8 Object Detection (Localization)")
    det_res = run_yolo_detection(image)

    if det_res.get("available"):
        d_col1, d_col2 = st.columns([1, 1])
        with d_col1:
            if det_res.get("annotated_image"):
                st.image(det_res["annotated_image"], caption="YOLOv8 Detection Overlays", use_container_width=True)
        with d_col2:
            st.success(f"Detected {det_res.get('num_detections', 0)} region(s).")
            st.json(det_res.get("detections", []))
    else:
        st.info("ℹ️ Detection model not trained or unavailable. Classification dataset does not contain bounding-box annotations.")

    st.markdown("---")

    # 5. Physical Measurement & Calibrations
    st.subheader("📐 Physical Measurements & Calibrations")
    meas = calculate_region_measurement(
        width_px=30.0,
        height_px=25.0,
        pixel_spacing_mm=meta.get("pixel_spacing_mm"),
        slice_thickness_mm=meta.get("slice_thickness_mm"),
    )
    st.write(f"- **Measurement Status:** `{meas['measurement_status']}`")
    for lim in meas.get("limitations", []):
        st.caption(f"• {lim}")

    st.markdown("---")

    # 6. Save Examination & PDF Generation
    st.subheader("💾 Examination Archiving & PDF Generation")
    exam_id = st.text_input("Examination ID", value=str(uuid.uuid4())[:8].upper())
    clinical_notes = st.text_area("Clinical / Research Notes", placeholder="Enter notes or observational findings...")

    save_c1, save_c2 = st.columns(2)
    with save_c1:
        if st.button("💾 Save Record to Database", use_container_width=True):
            save_examination(
                exam_id=exam_id,
                predictions=clf_results,
                ensemble=ens_res,
                detection=det_res,
                filename=uploaded_file.name,
                image_format=meta.get("format"),
                is_dicom=meta.get("is_dicom", False),
                pixel_spacing_mm=meta.get("pixel_spacing_mm"),
                measurement_status=meas.get("measurement_status"),
                notes=clinical_notes,
            )
            st.success(f"Examination #{exam_id} saved to research database successfully!")

    with save_c2:
        try:
            pdf_path = generate_pdf_report(
                exam_id=exam_id,
                predictions=clf_results,
                ensemble=ens_res,
                detection=det_res,
                image_metadata=meta,
                measurement_info=meas,
            )
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📄 Download PDF Research Report",
                    data=pdf_file.read(),
                    file_name=f"KidneyStone_Report_{exam_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Failed to generate PDF: {str(e)}")

else:
    st.info("👆 Please upload a patient CT scan slice (PNG, JPG, JPEG, or DICOM) above to initiate multi-model inference.")

render_disclaimer()
