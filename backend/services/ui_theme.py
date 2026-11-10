"""
UI Theme and Modern Styling Helper for Streamlit
Injects customized CSS, modern typography, glassmorphism cards,
gradient status badges, and medical disclaimers.
"""
import streamlit as st


def inject_custom_css():
    """Inject responsive, modern CSS design tokens into the Streamlit app."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }

    /* Custom Gradient Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F2027 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }

    .hero-title {
        color: #F8FAFC !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        margin-bottom: 6px !important;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94A3B8 !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        margin-bottom: 0px !important;
    }

    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 20px -8px rgba(14, 165, 233, 0.25);
    }

    .metric-title {
        color: #94A3B8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #F1F5F9;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .metric-subtitle {
        color: #64748B;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Model Prediction Card */
    .model-card {
        background: #1E293B;
        border-radius: 14px;
        border: 1px solid #334155;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .model-card:hover {
        border-color: #38BDF8;
    }

    .model-badge-stone {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.02em;
    }

    .model-badge-nonstone {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.02em;
    }

    .model-badge-unavailable {
        background: #475569;
        color: #CBD5E1;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }

    /* Medical Disclaimer Alert */
    .medical-disclaimer {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 14px 18px;
        color: #FCA5A5;
        font-size: 12px;
        line-height: 1.5;
        margin-top: 18px;
        margin-bottom: 18px;
    }

    /* Progress bar custom styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        border-radius: 9999px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_hero_banner(title: str, subtitle: str, badge: str = "RESEARCH SYSTEM"):
    """Render a modern hero header banner."""
    html = f"""
    <div class="hero-banner">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 2px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700; letter-spacing: 0.05em;">{badge}</span>
            <span style="color: #64748B; font-size: 12px; font-family: monospace;">PyTorch 2.6 • CUDA/CPU Auto-Fallback</span>
        </div>
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_disclaimer():
    """Render mandatory scientific research and clinical disclaimer."""
    html = """
    <div class="medical-disclaimer">
        <b>MANDATORY RESEARCH NOTICE:</b> This software is an academic engineering and deep learning evaluation prototype.
        Model probabilities, Grad-CAM heatmaps, and ensemble aggregations do NOT represent clinical diagnosis or validated radiologist impressions.
        Do not use this application as a medical diagnostic tool or for clinical treatment decisions.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
