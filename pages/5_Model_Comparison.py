"""
Model Comparison Page
Evaluates and benchmarks trained models strictly on the held-out test set,
displaying confusion matrices, sensitivity, specificity, ROC-AUC, latency, and model size.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from config.settings import CLASS_NAMES
from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from backend.services.classification import is_model_available, SUPPORTED_ARCHITECTURES
from training.evaluate_models import (
    evaluate_all_available_models,
    load_evaluation_results,
)

st.set_page_config(page_title="Model Comparison | Kidney Stone AI", page_icon="📈", layout="wide")
inject_custom_css()

render_hero_banner(
    title="Held-Out Test Set Model Comparison & Benchmarking",
    subtitle="Rigorous clinical metric evaluation (Accuracy, Sensitivity, Specificity, F1, ROC-AUC) strictly on isolated test patients.",
    badge="EVALUATION BENCHMARK",
)

col_actions = st.columns([1, 1, 2])
with col_actions[0]:
    eval_btn = st.button("🔄 Evaluate All Models on Test Set", type="primary", use_container_width=True)
with col_actions[1]:
    sample_cap = st.selectbox(
        "Evaluation Slices",
        options=[None, 100, 200],
        index=0,
        format_func=lambda x: "All Test Patients" if x is None else f"Cap at {x} (Fast)",
    )

if eval_btn:
    with st.spinner("Computing inferences across held-out test partition..."):
        try:
            results = evaluate_all_available_models(max_samples=sample_cap)
            st.session_state["eval_results"] = results
            st.success("Test evaluation finished successfully!")
        except Exception as e:
            st.error(f"Evaluation error: {str(e)}")

eval_results = st.session_state.get("eval_results") or load_evaluation_results()

if not eval_results or not eval_results.get("models"):
    st.info("No held-out evaluation results on disk yet. Click **Evaluate All Models on Test Set** above to generate metrics.")
    render_disclaimer()
    st.stop()

# 1. Comparative Metrics Table
st.subheader("📊 Primary Clinical Evaluation Metrics (Test Set)")
models_data = eval_results.get("models", {})
ensemble_data = eval_results.get("ensemble")

table_rows = []
for arch, data in models_data.items():
    if data.get("evaluated"):
        m = data["metrics"]
        table_rows.append({
            "Architecture": arch.upper().replace("_", " "),
            "Accuracy": f"{m['accuracy'] * 100:.1f}%",
            "Sensitivity (Stone)": f"{m['recall_sensitivity'] * 100:.1f}%",
            "Specificity (Non-Stone)": f"{m['specificity'] * 100:.1f}%",
            "Precision": f"{m['precision'] * 100:.1f}%",
            "F1-Score": f"{m['f1_score']:.3f}",
            "ROC-AUC": f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "N/A",
            "Avg Latency (ms)": data.get("avg_latency_ms", "N/A"),
            "Size (MB)": data.get("model_size_mb", "N/A"),
            "Samples": data.get("test_samples", 0),
        })
    else:
        table_rows.append({
            "Architecture": arch.upper().replace("_", " "),
            "Accuracy": "Not Evaluated",
            "Sensitivity (Stone)": "N/A",
            "Specificity (Non-Stone)": "N/A",
            "Precision": "N/A",
            "F1-Score": "N/A",
            "ROC-AUC": "N/A",
            "Avg Latency (ms)": "N/A",
            "Size (MB)": "N/A",
            "Samples": 0,
        })

if ensemble_data and ensemble_data.get("evaluated"):
    em = ensemble_data["metrics"]
    table_rows.append({
        "Architecture": "ENSEMBLE (AVERAGED)",
        "Accuracy": f"{em['accuracy'] * 100:.1f}%",
        "Sensitivity (Stone)": f"{em['recall_sensitivity'] * 100:.1f}%",
        "Specificity (Non-Stone)": f"{em['specificity'] * 100:.1f}%",
        "Precision": f"{em['precision'] * 100:.1f}%",
        "F1-Score": f"{em['f1_score']:.3f}",
        "ROC-AUC": f"{em['roc_auc']:.3f}" if em.get("roc_auc") is not None else "N/A",
        "Avg Latency (ms)": "Combined",
        "Size (MB)": "Combined",
        "Samples": ensemble_data.get("test_samples", 0),
    })

st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

# 2. Confusion Matrices
st.markdown("---")
st.subheader("🎯 Confusion Matrices (Held-Out Test Set)")

cm_cols = st.columns(len([k for k, v in models_data.items() if v.get("evaluated")]) or 1)
col_idx = 0

for arch, data in models_data.items():
    if data.get("evaluated"):
        with cm_cols[col_idx]:
            cm_info = data["metrics"]["confusion_matrix"]
            cm_mat = np.array(cm_info["matrix"])

            fig = px.imshow(
                cm_mat,
                text_auto=True,
                labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Cases"),
                x=CLASS_NAMES,
                y=CLASS_NAMES,
                color_continuous_scale="Blues",
                title=f"{arch.upper()}",
            )
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            col_idx += 1

# 3. Bar Chart Comparison
st.markdown("---")
st.subheader("📈 Metric Comparison Visualization")

chart_metrics = []
for r in table_rows:
    if r["Accuracy"] != "Not Evaluated":
        chart_metrics.append({
            "Architecture": r["Architecture"],
            "Accuracy": float(r["Accuracy"].replace("%", "")),
            "Sensitivity": float(r["Sensitivity (Stone)"].replace("%", "")),
            "Specificity": float(r["Specificity (Non-Stone)"].replace("%", "")),
        })

if chart_metrics:
    df_chart = pd.DataFrame(chart_metrics)
    df_melted = df_chart.melt(id_vars=["Architecture"], value_vars=["Accuracy", "Sensitivity", "Specificity"], var_name="Metric", value_name="Score (%)")

    fig_bar = px.bar(
        df_melted,
        x="Architecture",
        y="Score (%)",
        color="Metric",
        barmode="group",
        color_discrete_sequence=["#38BDF8", "#EF4444", "#10B981"],
        title="Comparison of Accuracy, Sensitivity (Stone), and Specificity (Non-Stone)",
    )
    fig_bar.update_layout(height=400, yaxis_range=[0, 105])
    st.plotly_chart(fig_bar, use_container_width=True)

render_disclaimer()
