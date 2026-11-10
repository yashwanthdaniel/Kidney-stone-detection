"""
Explainable AI (Grad-CAM) Page
Generates visual explanations of feature activations for MobileNetV2, ResNet50,
and EfficientNet-B0 using genuine gradient-weighted class activation mapping.
"""
import streamlit as st
from PIL import Image

from config.settings import CLASS_NAMES
from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from backend.services.image_loader import load_image
from backend.services.classification import is_model_available, SUPPORTED_ARCHITECTURES
from backend.services.explainability import generate_gradcam_explanation

st.set_page_config(page_title="Explainable AI | Kidney Stone AI", page_icon="💡", layout="wide")
inject_custom_css()

render_hero_banner(
    title="Explainable AI: Gradient-Weighted Class Activation Mapping",
    subtitle="Visualize salient convolutional regions influencing deep learning stone predictions.",
    badge="INTERPRETABILITY",
)

uploaded_file = st.file_uploader(
    "Upload Patient CT Slice for Heatmap Analysis",
    type=["png", "jpg", "jpeg", "dcm", "dicom"],
)

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    selected_arch = st.selectbox(
        "Classifier Architecture",
        SUPPORTED_ARCHITECTURES,
        format_func=lambda x: x.upper().replace("_", " "),
    )
with col_ctrl2:
    target_class = st.selectbox("Target Class for Activation Map", CLASS_NAMES, index=1)
with col_ctrl3:
    alpha = st.slider("Heatmap Blending Alpha", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

if uploaded_file is not None:
    try:
        image, meta = load_image(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read image: {str(e)}")
        st.stop()

    if not is_model_available(selected_arch):
        st.error(f"Selected model **{selected_arch.upper()}** has not been trained yet. Please train the model in the Training Center first.")
        render_disclaimer()
        st.stop()

    with st.spinner(f"Computing Grad-CAM gradients on {selected_arch.upper()}..."):
        cam_res = generate_gradcam_explanation(
            image=image,
            architecture=selected_arch,
            target_class_name=target_class,
            alpha=alpha,
        )

    if cam_res.get("success"):
        st.success(f"Grad-CAM generated successfully for **{target_class}** activation using **{selected_arch.upper()}**.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("1. Original CT Scan")
            st.image(image, caption="Original CT slice", use_container_width=True)
        with c2:
            st.subheader("2. Grad-CAM Heatmap")
            st.image(cam_res["heatmap_image"], caption=f"Activations: {target_class}", use_container_width=True)
        with c3:
            st.subheader("3. Heatmap Overlay")
            st.image(cam_res["overlay_image"], caption="Fused visualization", use_container_width=True)

        st.caption(cam_res["disclaimer"])
    else:
        st.error(f"Grad-CAM generation failed: {cam_res.get('error')}")

else:
    st.info("Upload a CT scan above to view Grad-CAM explainability heatmaps.")

render_disclaimer()
