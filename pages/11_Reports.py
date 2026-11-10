"""
Reports Archive & Export Page
Lists generated PDF reports on disk, provides on-demand report compilation,
and enforces academic research disclaimers.
"""
import streamlit as st
from pathlib import Path
from datetime import datetime

from config.settings import REPORTS_DIR
from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from backend.database.db_manager import get_all_examinations
from backend.reporting.pdf_generator import generate_pdf_report

st.set_page_config(page_title="Reports | Kidney Stone AI", page_icon="📄", layout="wide")
inject_custom_css()

render_hero_banner(
    title="Clinical Research Reports & PDF Documentation",
    subtitle="Audit, compile, and download publication-ready scientific reports generated via ReportLab.",
    badge="REPORT ARCHIVE",
)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
pdf_files = sorted(list(REPORTS_DIR.glob("*.pdf")), key=lambda x: x.stat().st_mtime, reverse=True)

st.subheader(f"📑 Stored PDF Reports ({len(pdf_files)} files)")

if pdf_files:
    for p in pdf_files:
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.markdown(f"**{p.name}**")
            st.caption(f"Modified: {datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')} | Size: {round(p.stat().st_size / 1024, 1)} KB")
        with c2:
            with open(p, "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=f.read(),
                    file_name=p.name,
                    mime="application/pdf",
                    key=f"dl_{p.name}",
                )
        with c3:
            if st.button("🗑️ Delete", key=f"del_{p.name}"):
                p.unlink(missing_ok=True)
                st.rerun()
        st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid #334155;'>", unsafe_allow_html=True)
else:
    st.info("No generated PDF reports currently in the `reports/` folder. Generate reports from the **CT Image Analysis** page.")

render_disclaimer()
