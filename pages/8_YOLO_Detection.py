"""
YOLOv8 Object Detection & Localization Module
Comprehensive interface for YOLOv8 kidney stone detection, dataset validation,
annotation guidance (LabelImg/CVAT), CUDA GPU training, and independent test evaluation.
"""
import streamlit as st
from pathlib import Path
from PIL import Image
import json
import pandas as pd
import torch

from config.settings import (
    DETECTION_MODELS_DIR,
    YOLO_DATA_YAML,
    YOLO_REPORTS_DIR,
    get_hardware_info,
)
from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from backend.services.image_loader import load_image
from backend.services.detection import (
    is_detector_available,
    run_yolo_detection,
    get_detection_checkpoint_path,
)
from training.validate_yolo_dataset import validate_yolo_dataset
from training.prepare_and_train_yolo import prepare_yolo_dataset
from training.train_yolo import train_yolo_detector, check_cuda_requirement
from training.evaluate_yolo import evaluate_yolo_on_test_set

st.set_page_config(page_title="YOLOv8 Detection | Kidney Stone AI", page_icon="🎯", layout="wide")
inject_custom_css()

render_hero_banner(
    title="YOLOv8 Kidney Stone Object Detection & Localization",
    subtitle="Bounding-box localization network strictly separated from classification, powered by CUDA GPU acceleration.",
    badge="OBJECT DETECTION",
)

tab_inf, tab_data, tab_train, tab_eval = st.tabs([
    "🔍 1. Detection Inference",
    "📁 2. Dataset & Annotations",
    "⚙️ 3. GPU Model Training",
    "📊 4. Test Evaluation & Metrics",
])

# =============================================================================
# TAB 1: DETECTION INFERENCE
# =============================================================================
with tab_inf:
    st.subheader("Run YOLOv8 Kidney Stone Localization")
    detector_ready = is_detector_available()

    if not detector_ready:
        st.warning(
            "⚠️ **YOLOv8 Checkpoint Not Found (`models/detection/best.pt`)**\n\n"
            "The detection model has not been trained yet. To run object detection:\n"
            "1. Switch to **Tab 2 (Dataset & Annotations)** to verify or import bounding-box annotations.\n"
            "2. Switch to **Tab 3 (GPU Model Training)** to train YOLOv8 on your NVIDIA RTX 5060 GPU."
        )

    col_up, col_ctrl = st.columns([2, 1])
    with col_up:
        uploaded_det_file = st.file_uploader(
            "Upload Patient CT Scan Slice",
            type=["png", "jpg", "jpeg", "dcm", "dicom"],
            key="yolo_page_uploader",
        )
    with col_ctrl:
        conf_thresh = st.slider("Detection Confidence Threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        iou_thresh = st.slider("NMS IoU Threshold", min_value=0.10, max_value=0.80, value=0.45, step=0.05)

    if uploaded_det_file is not None:
        try:
            det_img, det_meta = load_image(uploaded_det_file)
        except Exception as e:
            st.error(f"Error reading image: {str(e)}")
            st.stop()

        col_orig, col_box = st.columns(2)
        with col_orig:
            st.subheader("Input CT Image")
            st.image(det_img, caption=f"Scan: {uploaded_det_file.name}", use_container_width=True)
            st.caption(f"Dimensions: {det_img.width} x {det_img.height} px | Format: {det_meta.get('format', '2D')}")

        with col_box:
            st.subheader("YOLOv8 Stone Localization")
            if detector_ready:
                with st.spinner("Executing YOLOv8 inference on GPU..."):
                    res = run_yolo_detection(
                        det_img,
                        conf_threshold=conf_thresh,
                        iou_threshold=iou_thresh,
                    )

                if res.get("available") and res.get("annotated_image"):
                    st.image(res["annotated_image"], caption="YOLOv8 Localization Overlay", use_container_width=True)

                    n_det = res.get("num_detections", 0)
                    lat = res.get("inference_time_ms", 0)

                    if n_det > 0:
                        st.success(f"🎯 **{n_det} Stone Region(s) Detected** | Latency: `{lat} ms` | Device: `{res.get('device')}`")
                    else:
                        st.info(f"ℹ️ No stone calcifications identified above {round(conf_thresh*100)}% confidence threshold.")

                    # Coordinates table
                    detections = res.get("detections", [])
                    if detections:
                        st.markdown("#### Detected Stone Coordinates")
                        df_rows = []
                        for i, d in enumerate(detections, 1):
                            bx = d["box_xyxy_pixels"]
                            ny = d["box_normalized_yolo"]
                            df_rows.append({
                                "Stone #": i,
                                "Confidence": f"{round(d['confidence']*100, 1)}%",
                                "Width (px)": d["width_px"],
                                "Height (px)": d["height_px"],
                                "Box [x1, y1, x2, y2]": f"[{bx[0]}, {bx[1]}, {bx[2]}, {bx[3]}]",
                                "YOLO [xc, yc, w, h]": f"[{ny[0]}, {ny[1]}, {ny[2]}, {ny[3]}]",
                            })
                        st.dataframe(pd.DataFrame(df_rows), use_container_width=True)
                else:
                    st.error(f"Inference error: {res.get('status')}")
            else:
                st.info("Train the YOLOv8 model in Tab 3 to visualize bounding boxes.")

# =============================================================================
# TAB 2: DATASET & ANNOTATIONS
# =============================================================================
with tab_data:
    st.subheader("YOLO Dataset Structure & Bounding-Box Annotations")

    st.markdown("""
    YOLOv8 object detection requires **bounding-box coordinate labels**, unlike classification which only uses folder-level labels.
    - **Stone Images**: Require `.txt` files containing normalized coordinates: `0 <x_center> <y_center> <width> <height>`.
    - **Non-Stone Images**: Use empty `.txt` files (0 bytes) as negative background samples to prevent false positives.
    - **Patient Isolation**: All slices belonging to a patient must reside strictly in either train, validation, or test.
    """)

    st.markdown("---")
    st.markdown("### 📝 How to Annotate Using LabelImg or CVAT")

    with st.expander("📖 Step-by-Step Annotation Guide (LabelImg / CVAT)", expanded=False):
        st.markdown("""
        #### Tool 1: LabelImg (Fastest for local use)
        1. Install via terminal:
           ```bash
           pip install labelImg
           labelImg
           ```
        2. In LabelImg, click **'Open Dir'** and select `dataset/Original/Stone`.
        3. Click **'Change Save Dir'** and create/select a folder `dataset/labels`.
        4. In the left toolbar, click the format button until it displays **'YOLO'** (not PascalVOC).
        5. Draw a tight rectangular bounding box around each visible kidney stone calcification.
        6. Label the class as **`kidney_stone`** (class ID 0).
        7. Save each image (hotkey `Ctrl + S`, next image `D`).

        #### Tool 2: CVAT (Computer Vision Annotation Tool - Web-based)
        1. Open [cvat.ai](https://www.cvat.ai) or run locally via Docker.
        2. Create a new task with label `kidney_stone`.
        3. Upload your stone CT images.
        4. Draw bounding boxes around calcifications.
        5. Export annotations as **YOLO 1.1** format.
        6. Extract the generated `.txt` files into `dataset/labels/`.
        """)

    st.markdown("---")
    st.subheader("Validate Dataset & Patient Isolation")

    c_v1, c_v2 = st.columns([1, 1])
    with c_v1:
        if st.button("🔍 Run Dataset Validation Check", type="primary", use_container_width=True):
            with st.spinner("Validating YOLO dataset structure, coordinate geometry, and patient isolation..."):
                val_rep = validate_yolo_dataset(verify_image_files=False)

            if val_rep["valid"]:
                st.success("✅ **Dataset is Fully Valid for YOLOv8 Training!**")
            else:
                st.error("❌ **Dataset Validation Failed**")

            st.write(f"- **Total Images:** `{val_rep['total_images']}`")
            st.write(f"- **Total Stone Bounding Boxes:** `{val_rep['total_boxes']}`")
            st.write(f"- **Positive (Stone) Slices:** `{val_rep['positive_images']}`")
            st.write(f"- **Negative (Non-Stone) Slices:** `{val_rep['negative_images']}`")

            iso = val_rep["patient_isolation"]
            if iso["verified"]:
                st.success(f"✅ **Zero Patient Leakage:** Train: {iso['train_patients']} pats, Val: {iso['val_patients']} pats, Test: {iso['test_patients']} pats.")
            else:
                st.error(f"❌ **Data Leakage Detected:** Overlaps: Train-Val: {iso['overlap_train_val']}, Train-Test: {iso['overlap_train_test']}, Val-Test: {iso['overlap_val_test']}")

            if val_rep["errors"]:
                st.markdown("#### Errors to Resolve:")
                for e in val_rep["errors"][:10]:
                    st.error(e)
            if val_rep["warnings"]:
                for w in val_rep["warnings"]:
                    st.warning(w)

    with c_v2:
        custom_anno_path = st.text_input("Custom Annotations Folder (Optional)", placeholder="dataset/labels")
        if st.button("📦 Prepare & Assemble YOLO Dataset Structure", use_container_width=True):
            with st.spinner("Assembling train/val/test splits, copying images, pairing annotations, and creating data.yaml..."):
                p_res = prepare_yolo_dataset(
                    annotation_source_dir=Path(custom_anno_path) if custom_anno_path else None
                )
            if p_res["success"]:
                st.success(f"YOLO dataset assembled successfully at `{p_res['dataset_root']}`!")
            else:
                st.warning(
                    f"Dataset structure prepared, but {p_res['missing_stone_annotations_count']} stone image(s) lack bounding-box annotations.\n\n"
                    "Place your `.txt` files in `dataset/labels/` and click prepare again."
                )

# =============================================================================
# TAB 3: GPU MODEL TRAINING
# =============================================================================
with tab_train:
    st.subheader("YOLOv8 Detection Model Training on NVIDIA CUDA GPU")

    # Hardware check
    cuda_available, cuda_msg = check_cuda_requirement()
    if cuda_available:
        st.success(f"🖥️ **Hardware Ready:** {cuda_msg}")
    else:
        st.error(f"⚠️ **Hardware Warning:** {cuda_msg}")

    st.markdown("---")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        tr_epochs = st.number_input("Training Epochs", min_value=5, max_value=300, value=100, step=5)
        tr_batch = st.selectbox("Batch Size (GPU VRAM Optimized)", [8, 16, 32], index=1, help="Batch 16 recommended for RTX 5060 8GB.")
        tr_imgsz = st.selectbox("Input Image Size", [512, 640], index=1)
    with col_t2:
        tr_opt = st.selectbox("Optimizer", ["AdamW", "SGD", "auto"], index=0)
        tr_patience = st.number_input("Early Stopping Patience (Epochs)", min_value=5, max_value=50, value=15)
        tr_resume = st.checkbox("Resume from last checkpoint (last.pt) if interrupted", value=False)

    st.markdown("---")
    st.info("💡 **Pre-training Validation:** Training will only begin if bounding-box annotations are verified and no patient leakage exists.")

    if st.button("🚀 Start YOLOv8 GPU Training", type="primary", use_container_width=True):
        if not cuda_available:
            st.error("Cannot train: NVIDIA CUDA GPU is required. Check your drivers.")
        else:
            with st.spinner("Validating dataset and launching YOLOv8 training on CUDA GPU..."):
                t_res = train_yolo_detector(
                    epochs=tr_epochs,
                    batch_size=tr_batch,
                    img_size=tr_imgsz,
                    patience=tr_patience,
                    optimizer=tr_opt,
                    resume=tr_resume,
                )

            if t_res.get("success"):
                st.success(f"🎉 **Training Completed in {round(t_res['duration_seconds']/60, 1)} minutes!**")
                st.write(f"- **Best Checkpoint:** `{t_res['best_checkpoint']}`")
                st.write(f"- **Last Checkpoint:** `{t_res['last_checkpoint']}`")
            else:
                st.error(f"Training could not proceed: {t_res.get('error')}")

# =============================================================================
# TAB 4: TEST EVALUATION & METRICS
# =============================================================================
with tab_eval:
    st.subheader("Independent Test Set Evaluation")
    st.caption("Evaluates models/detection/best.pt on the unseen test partition. Strictly authentic calculations without fabricated values.")

    best_exists = is_detector_available()

    if not best_exists:
        st.warning("No trained checkpoint found at `models/detection/best.pt`. Train the model first in Tab 3.")
    else:
        st.success("Trained checkpoint `models/detection/best.pt` is available for independent testing.")

        c_eval_btn, c_eval_opts = st.columns([1, 1])
        with c_eval_btn:
            run_eval = st.button("📊 Evaluate on Independent Test Set", type="primary", use_container_width=True)
        with c_eval_opts:
            eval_conf = st.slider("Evaluation Confidence", min_value=0.10, max_value=0.90, value=0.25, step=0.05, key="eval_conf")

        if run_eval:
            with st.spinner("Evaluating model on independent test partition..."):
                try:
                    eval_res = evaluate_yolo_on_test_set(conf_threshold=eval_conf)
                    st.success("Test set evaluation completed successfully!")

                    # Metrics display
                    ec1, ec2, ec3, ec4 = st.columns(4)
                    with ec1:
                        st.metric("Precision", f"{eval_res['precision']*100:.1f}%")
                    with ec2:
                        st.metric("Recall", f"{eval_res['recall']*100:.1f}%")
                    with ec3:
                        st.metric("mAP@50", f"{eval_res['mAP_50']*100:.1f}%")
                    with ec4:
                        st.metric("mAP@50-95", f"{eval_res['mAP_50_95']*100:.1f}%")

                    st.markdown(f"**Inference Latency:** `{eval_res['inference_latency_ms']['total_latency']} ms/slice`")

                    # Display test plots if present
                    plots = list(YOLO_REPORTS_DIR.glob("*.png"))
                    if plots:
                        st.markdown("#### Evaluation Visualizations")
                        p_cols = st.columns(min(3, len(plots)))
                        for idx, p_path in enumerate(plots[:6]):
                            with p_cols[idx % len(p_cols)]:
                                st.image(str(p_path), caption=p_path.name, use_container_width=True)

                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")

render_disclaimer()
