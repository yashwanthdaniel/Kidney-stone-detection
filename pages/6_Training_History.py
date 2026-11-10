"""
Training History Page
Visualizes epoch-by-epoch loss curves and accuracy trajectories from real saved history logs.
Strictly avoids placeholder or simulated data.
"""
import streamlit as st
import plotly.graph_objects as go

from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from backend.services.classification import SUPPORTED_ARCHITECTURES
from training.train_classifier import load_training_history

st.set_page_config(page_title="Training History | Kidney Stone AI", page_icon="📜", layout="wide")
inject_custom_css()

render_hero_banner(
    title="Epoch-Level Training Trajectories",
    subtitle="Inspect training vs. validation loss convergence and accuracy dynamics from genuine training runs.",
    badge="TRAINING TELEMETRY",
)

selected_arch = st.selectbox(
    "Select Architecture",
    SUPPORTED_ARCHITECTURES,
    format_func=lambda x: x.upper().replace("_", " "),
)

history = load_training_history(selected_arch)

if not history or not history.get("epochs"):
    st.info(f"No training history found on disk for **{selected_arch.upper()}**. Train this model in the Training Center first to record epoch dynamics.")
    render_disclaimer()
    st.stop()

# 1. Summary Cards
h1, h2, h3, h4 = st.columns(4)
with h1:
    st.metric("Total Epochs Trained", len(history["epochs"]))
with h2:
    st.metric("Optimal Epoch", history.get("best_epoch", 0))
with h3:
    st.metric("Best Validation Loss", history.get("best_val_loss", "N/A"))
with h4:
    st.metric("Best Validation Accuracy", f"{history.get('best_val_acc', 0)}%")

st.markdown("---")

# 2. Dual Plotly Curves: Loss and Accuracy
col_loss, col_acc = st.columns(2)

epochs_x = history["epochs"]

with col_loss:
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs_x, y=history["train_loss"],
        mode="lines+markers", name="Train Loss",
        line=dict(color="#38BDF8", width=2.5),
    ))
    fig_loss.add_trace(go.Scatter(
        x=epochs_x, y=history["val_loss"],
        mode="lines+markers", name="Validation Loss",
        line=dict(color="#F43F5E", width=2.5, dash="dash"),
    ))
    fig_loss.update_layout(
        title="Training vs. Validation Loss (CrossEntropy)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(x=0.02, y=0.98),
        height=400,
    )
    st.plotly_chart(fig_loss, use_container_width=True)

with col_acc:
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=epochs_x, y=history["train_acc"],
        mode="lines+markers", name="Train Accuracy",
        line=dict(color="#10B981", width=2.5),
    ))
    fig_acc.add_trace(go.Scatter(
        x=epochs_x, y=history["val_acc"],
        mode="lines+markers", name="Validation Accuracy",
        line=dict(color="#A855F7", width=2.5, dash="dash"),
    ))
    fig_acc.update_layout(
        title="Training vs. Validation Accuracy (%)",
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 105],
        legend=dict(x=0.02, y=0.10),
        height=400,
    )
    st.plotly_chart(fig_acc, use_container_width=True)

st.caption(f"Training session executed on device: `{history.get('device', 'cpu')}` | Recorded: `{history.get('timestamp')}`")

render_disclaimer()
