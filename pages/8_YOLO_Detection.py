"""
YOLOv8 Object Detection & Localization Page
Handles YOLO bounding-box detection, dataset YAML verification,
and training when localization labels exist. Strictly avoids inventing fake bounding boxes.
"""
import streamlit as st
from pathlib import Path
from PIL import Image

from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from backend.services.image_loader import load_image
from backend.services.detection import (
    is_detector_available,
    run_yolo_detection,
    get_detection_checkpoint_path,
)
from training.train_yolo import validate_yolo_yaml, train_yolo_model

st.set_page_config(page_title="YOLO Detection | Kidney Stone AI", page_icon="🎯", layout="wide")
inject_custom_css()

render_hero_banner(
    title="YOLOv8 Object Detection & Stone Localization",
    subtitle="Bounding-box stone detection service strictly separated from whole-slice classification.",
    badge="OBJECT DETECTION",
)

tab_inf, tab_train = st.tabs(["🔍 Detection Inference", "🛠️ Detection Model Training"])

with tab_inf:
    st.subheader("Run YOLOv8 Stone Localization")
    detector_ready = is_detector_available()

    if not detector_ready:
        st.info(
            "ℹ️ **Detection Model Status:** No trained YOLOv8 checkpoint found at `models/detection/best.pt`. "
            "The current dataset provides whole-slice image classification labels (Stone / Non-Stone) "
            "and does not contain bounding-box localization coordinates. "
            "To train YOLOv8, provide a dataset formatted with bounding-box annotations in the Training tab."
        )

    uploaded_det_file = st.file_uploader(
        "Upload CT Image for Bounding Box Detection",
        type=["png", "jpg", "jpeg", "dcm"],
        key="yolo_uploader",
    )

    conf_thresh = st.slider("Detection Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)

    if uploaded_det_file is not None:
        try:
            det_img, det_meta = load_image(uploaded_det_file)
        except Exception as e:
            st.error(f"Error reading image: {str(e)}")
            st.stop()

        col_orig, col_box = st.columns(2)
        with col_orig:
            st.subheader("Input CT Image")
            st.image(det_img, caption=uploaded_det_file.name, use_container_width=True)

        with col_box:
            st.subheader("Detected Bounding Boxes")
            if detector_ready:
                res = run_yolo_detection(det_img, conf_threshold=conf_thresh)
                if res.get("available") and res.get("annotated_image"):
                    st.image(res["annotated_image"], caption="YOLOv8 Stone Localization", use_container_width=True)
                    st.json(res.get("detections", []))
                else:
                    st.warning("No kidney stones localized above the confidence threshold.")
            else:
                st.warning("Detection model not trained or unavailable. Please train a detector with valid bounding boxes.")

with tab_train:
    st.subheader("Train YOLOv8 on Custom Annotated Dataset")
    st.caption("Provide a valid dataset YAML pointing to images and bounding-box labels (e.g. YOLO format TXT annotations).")

    yaml_input = st.text_input("Dataset YAML Path", value="data/yolo_dataset.yaml")
    y_epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20)
    y_batch = st.selectbox("Batch Size", [8, 16, 32], index=1)

    if st.button("Validate YAML & Train YOLOv8", type="primary"):
        val_res = validate_yolo_yaml(Path(yaml_input))
        if not val_res["valid"]:
            st.error(f"Dataset YAML validation failed: {val_res['error']}")
        else:
            st.success("Dataset YAML validated successfully! Starting training...")
            with st.spinner("Training YOLOv8..."):
                t_res = train_yolo_model(yaml_path=yaml_input, epochs=y_epochs, batch_size=y_batch)
                if t_res.get("success"):
                    st.success(f"YOLOv8 trained and saved to: {t_res['checkpoint_path']}")
                else:
                    st.error(f"Training failed: {t_res.get('error')}")

render_disclaimer()
