"""
Training Center Page: Multi-Model Deep Learning Training
Provides interactive controls to train MobileNetV2, ResNet50, and EfficientNet-B0
independently or sequentially with real-time epoch telemetry and early stopping.
"""
import streamlit as st
import time

from config.settings import (
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PATIENCE,
    DEFAULT_RANDOM_SEED,
    get_hardware_info,
    get_device,
)
from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from backend.services.classification import (
    is_model_available,
    SUPPORTED_ARCHITECTURES,
    get_checkpoint_path,
)
from training.split_dataset import load_split_manifest
from training.train_classifier import train_single_classifier, train_all_classifiers

st.set_page_config(page_title="Training Center | Kidney Stone AI", page_icon="🚀", layout="wide")
inject_custom_css()

render_hero_banner(
    title="Deep Learning Training Center",
    subtitle="Train MobileNetV2, ResNet50, and EfficientNet-B0 on patient-isolated clinical partitions.",
    badge="TRAINING PIPELINE",
)

# 1. Environment & Partitions Telemetry
manifest = load_split_manifest()
hw = get_hardware_info()

t1, t2, t3, t4 = st.columns(4)
with t1:
    st.metric("Compute Target", hw["device_selected"].upper(), f"PyTorch {hw['torch_version']}")
with t2:
    st.metric("Free Disk Space", f"{hw['disk_free_gb']} GB")
with t3:
    train_count = manifest["split_counts"]["train"]["total"] if manifest else 0
    st.metric("Train Samples", f"{train_count:,}")
with t4:
    val_count = manifest["split_counts"]["val"]["total"] if manifest else 0
    st.metric("Val Samples", f"{val_count:,}")

if not manifest:
    st.warning("⚠️ No dataset splits found! Please visit **Dataset Manager** to create train/val partitions before initiating training.")

# 2. Hyperparameter Settings
st.subheader("⚙️ Training Hyperparameters")
col_h1, col_h2, col_h3, col_h4 = st.columns(4)

with col_h1:
    epochs = st.number_input("Max Epochs", min_value=1, max_value=100, value=10, step=1)
    patience = st.number_input("Early Stopping Patience", min_value=1, max_value=20, value=3, step=1)

with col_h2:
    batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=2)
    learning_rate = st.select_slider("Learning Rate", options=[1e-5, 5e-5, 1e-4, 3e-4, 1e-3], value=1e-4)

with col_h3:
    random_seed = st.number_input("Random Seed", value=DEFAULT_RANDOM_SEED, step=1)
    max_train_samples = st.selectbox(
        "Training Sample Cap (Speed)",
        options=[None, 500, 1000, 2500, 5000, 10000],
        index=0,
        format_func=lambda x: "All Training Slices" if x is None else f"Cap at {x} slices (Fast Mode)",
    )

with col_h4:
    st.markdown("**Existing Trained Checkpoints:**")
    for arch in SUPPORTED_ARCHITECTURES:
        ready = is_model_available(arch)
        status_text = "✅ Ready" if ready else "❌ Not trained"
        st.write(f"- **{arch.upper()}**: {status_text}")

st.markdown("---")

# 3. Training Execution Controls
st.subheader("🚀 Training Controls")

if "is_training" not in st.session_state:
    st.session_state["is_training"] = False

btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)

target_arch = None
train_all = False

with btn_c1:
    if st.button("Train MobileNetV2", disabled=st.session_state["is_training"] or not manifest):
        target_arch = "mobilenet_v2"

with btn_c2:
    if st.button("Train ResNet50", disabled=st.session_state["is_training"] or not manifest):
        target_arch = "resnet50"

with btn_c3:
    if st.button("Train EfficientNet-B0", disabled=st.session_state["is_training"] or not manifest):
        target_arch = "efficientnet_b0"

with btn_c4:
    if st.button("⚡ Train All Models (Sequential)", type="primary", disabled=st.session_state["is_training"] or not manifest):
        train_all = True

# Progress containers
status_container = st.empty()
progress_bar = st.empty()
metrics_table_container = st.empty()

def update_ui_progress(info: dict):
    arch = info["architecture"]
    ep = info["epoch"]
    tot_ep = info["total_epochs"]
    status_container.info(
        f"**Training {arch.upper().replace('_', ' ')}** — Epoch {ep}/{tot_ep} | "
        f"Train Loss: `{info['train_loss']}` | Val Loss: `{info['val_loss']}` | "
        f"Val Acc: `{info['val_acc']}%` | Best Val Acc: `{info['best_val_acc']}%`"
    )
    progress_bar.progress(ep / tot_ep)

if target_arch or train_all:
    st.session_state["is_training"] = True
    try:
        if train_all:
            st.info("Starting sequential training for all three architectures...")
            results = train_all_classifiers(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=patience,
                random_seed=random_seed,
                max_train_samples=max_train_samples,
                progress_callback=update_ui_progress,
            )
            st.success("All models trained successfully!")
            st.json({k: {"best_val_acc": v["best_val_acc"], "best_epoch": v["best_epoch"]} for k, v in results.items()})
        else:
            st.info(f"Initiating training for {target_arch.upper()}...")
            res = train_single_classifier(
                architecture=target_arch,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=patience,
                random_seed=random_seed,
                max_train_samples=max_train_samples,
                progress_callback=update_ui_progress,
            )
            st.success(f"{target_arch.upper()} training complete! Best Validation Accuracy: {res['best_val_acc']}% at epoch {res['best_epoch']}.")
    except Exception as e:
        st.error(f"Training encountered an error: {str(e)}")
    finally:
        st.session_state["is_training"] = False

render_disclaimer()
