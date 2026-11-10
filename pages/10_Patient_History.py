"""
Patient & Research Examination History Page
Browses saved study examinations, review individual classifier votes,
inspects ensemble consensus, and re-downloads clinical reports.
"""
import streamlit as st
import pandas as pd

from backend.services.ui_theme import inject_custom_css, render_hero_banner, render_disclaimer
from backend.database.db_manager import get_all_examinations, get_examination_by_id
from backend.reporting.pdf_generator import generate_pdf_report

st.set_page_config(page_title="Patient History | Kidney Stone AI", page_icon="📋", layout="wide")
inject_custom_css()

render_hero_banner(
    title="Research Examination History & Case Records",
    subtitle="Audit persistent SQLite records of deep learning evaluations, calibrated measurements, and ensemble consensus.",
    badge="DATABASE AUDIT",
)

exams = get_all_examinations(limit=200)

if not exams:
    st.info("No examination records stored in SQLite database yet. Upload and analyze an image in **CT Image Analysis** to populate study history.")
    render_disclaimer()
    st.stop()

# Summary table
table_data = []
for ex in exams:
    table_data.append({
        "Examination ID": ex["id"],
        "Timestamp": ex["timestamp"][:19].replace("T", " "),
        "Filename": ex.get("filename", "N/A"),
        "Ensemble Class": ex.get("predicted_class", "N/A"),
        "Confidence": f"{round((ex.get('confidence') or 0.0) * 100, 1)}%",
        "DICOM Calibrated": "Yes" if ex.get("is_dicom") else "No",
        "Measurement": ex.get("measurement_status", "N/A"),
    })

st.dataframe(pd.DataFrame(table_data), use_container_width=True)

st.markdown("---")
st.subheader("🔍 Detailed Case Inspector")

selected_id = st.selectbox("Select Examination ID to Inspect", [e["id"] for e in exams])
record = get_examination_by_id(selected_id)

if record:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown(f"**Exam ID:** `{record['id']}`")
        st.markdown(f"**Timestamp:** `{record['timestamp']}`")
        st.markdown(f"**Original Image:** `{record.get('filename')}`")
        st.markdown(f"**DICOM Calibrated:** `{'Yes' if record.get('is_dicom') else 'No'}`")
        st.markdown(f"**Measurement Status:** `{record.get('measurement_status')}`")
        if record.get("notes"):
            st.info(f"**Notes:** {record['notes']}")

    with c2:
        st.markdown("### Model Predictions")
        preds = record.get("predictions", {})
        for m_name, p in preds.items():
            if p.get("available"):
                st.write(f"- **{m_name.upper()}**: {p.get('predicted_class')} ({round(p.get('confidence', 0)*100, 1)}% confidence, latency: {p.get('inference_time_ms', 0)} ms)")
            else:
                st.write(f"- **{m_name.upper()}**: Unavailable")

        ens = record.get("ensemble")
        if ens and ens.get("available"):
            st.markdown(f"**Ensemble Decision:** `{ens.get('predicted_class')}` ({round(ens.get('confidence', 0)*100, 1)}% confidence)")

    # Download PDF
    try:
        pdf_path = generate_pdf_report(
            exam_id=record["id"],
            predictions=record.get("predictions", {}),
            ensemble=record.get("ensemble"),
            detection=record.get("detection"),
            image_metadata={"format": record.get("image_format"), "is_dicom": bool(record.get("is_dicom"))},
            measurement_info={"measurement_status": record.get("measurement_status"), "limitations": []},
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                label=f"📄 Download PDF Report for #{record['id']}",
                data=f.read(),
                file_name=f"Report_{record['id']}.pdf",
                mime="application/pdf",
            )
    except Exception as e:
        st.caption(f"PDF generation note: {str(e)}")

render_disclaimer()
