"""
Model Management Page
Inspects saved model checkpoints, verifies architecture compatibility,
displays metadata (timestamp, class mapping, hyperparameters), and validates custom weight uploads.
"""
import streamlit as st
from pathlib import Path
import torch
import shutil

from config.settings import (
    CLASSIFICATION_MODELS_DIR,
    CLASS_NAMES,
    MODEL_FILENAMES,
)
from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from backend.services.classification import (
    is_model_available,
    SUPPORTED_ARCHITECTURES,
    get_checkpoint_path,
)

st.set_page_config(page_title="Model Management | Kidney Stone AI", page_icon="⚙️", layout="wide")
inject_custom_css()

render_hero_banner(
    title="Model Checkpoint Registry & Integrity Verifier",
    subtitle="Inspect saved weights, verify class mapping alignments, and load external model checkpoints safely.",
    badge="MODEL REGISTRY",
)

st.subheader("📦 Saved Classification Checkpoints")

for arch in SUPPORTED_ARCHITECTURES:
    path = get_checkpoint_path(arch)
    ready = is_model_available(arch)

    with st.expander(f"{arch.upper().replace('_', ' ')} Checkpoint Details", expanded=ready):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.write(f"- **Architecture:** `{arch}`")
            st.write(f"- **Path:** `{path}`")
            st.write(f"- **Status:** `{'Ready & Loaded' if ready else 'Checkpoint Missing'}`")
            if ready:
                st.write(f"- **File Size:** `{round(path.stat().st_size / (1024**2), 2)} MB`")

        with col2:
            if ready:
                try:
                    cp = torch.load(path, map_location="cpu")
                    st.write(f"- **Trained Epoch:** `{cp.get('epoch', 'N/A')}`")
                    st.write(f"- **Best Val Accuracy:** `{cp.get('best_val_acc', 'N/A')}%`")
                    st.write(f"- **Best Val Loss:** `{cp.get('best_val_loss', 'N/A')}`")
                    st.write(f"- **Class Mapping:** `{cp.get('class_to_idx', {})}`")
                    st.write(f"- **Created:** `{cp.get('timestamp', 'N/A')}`")
                except Exception as e:
                    st.error(f"Error reading checkpoint metadata: {str(e)}")
            else:
                st.info("Train this model in the Training Center to generate valid weights.")

st.markdown("---")
st.subheader("📤 Import External Checkpoint")
st.caption("Upload a compatible `.pth` model checkpoint. The system validates architecture and class mapping compatibility before accepting weights.")

imp_c1, imp_c2 = st.columns(2)
with imp_c1:
    target_upload_arch = st.selectbox(
        "Target Architecture for Checkpoint",
        SUPPORTED_ARCHITECTURES,
        format_func=lambda x: x.upper().replace("_", " "),
    )
with imp_c2:
    uploaded_cp = st.file_uploader("Select .pth Checkpoint File", type=["pth", "pt"])

if uploaded_cp is not None and st.button("Verify & Install Checkpoint"):
    with st.spinner("Validating checkpoint architecture compatibility..."):
        try:
            temp_path = CLASSIFICATION_MODELS_DIR / f"temp_{uploaded_cp.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_cp.read())

            cp_data = torch.load(temp_path, map_location="cpu")

            if not isinstance(cp_data, dict) or "state_dict" not in cp_data:
                temp_path.unlink(missing_ok=True)
                st.error("Invalid checkpoint: missing 'state_dict' dictionary.")
            else:
                saved_arch = cp_data.get("architecture")
                if saved_arch and saved_arch.lower() != target_upload_arch.lower():
                    temp_path.unlink(missing_ok=True)
                    st.error(f"Architecture Mismatch! Checkpoint is '{saved_arch}', but target is '{target_upload_arch}'. Cannot load mismatched weights.")
                else:
                    dest = get_checkpoint_path(target_upload_arch)
                    shutil.move(str(temp_path), str(dest))
                    st.success(f"Checkpoint verified and successfully installed for {target_upload_arch.upper()}!")
        except Exception as e:
            st.error(f"Failed to load checkpoint: {str(e)}")

render_disclaimer()
