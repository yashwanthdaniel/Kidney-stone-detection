"""
PDF Report Generator using ReportLab
Produces comprehensive, downloadable clinical research documentation
with individual classifier probabilities, ensemble outcomes, detection results,
calibrated measurements, and scientific disclaimers.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

from config.settings import REPORTS_DIR


def generate_pdf_report(
    exam_id: str,
    predictions: Dict[str, Dict[str, Any]],
    ensemble: Optional[Dict[str, Any]] = None,
    detection: Optional[Dict[str, Any]] = None,
    image_metadata: Optional[Dict[str, Any]] = None,
    measurement_info: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate professional PDF research report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path or (REPORTS_DIR / f"Report_{exam_id}.pdf")

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12,
    )
    heading2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#742A2A"),
        backColor=colors.HexColor("#FFF5F5"),
        borderColor=colors.HexColor("#FEB2B2"),
        borderWidth=1,
        borderPadding=6,
        spaceBefore=10,
        spaceAfter=8,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("AI-Assisted Kidney Stone Detection & Classification Report", title_style))
    story.append(Paragraph(f"Research Examination ID: {exam_id} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3182CE"), spaceAfter=10))

    # Study Information
    story.append(Paragraph("1. Image & Study Metadata", heading2_style))
    meta_data = [
        [Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Value</b>", body_style)],
        [Paragraph("Examination ID", body_style), Paragraph(exam_id, body_style)],
        [Paragraph("Modality", body_style), Paragraph(str(image_metadata.get("modality", "CT Scan") if image_metadata else "CT Scan"), body_style)],
        [Paragraph("Image Format", body_style), Paragraph(str(image_metadata.get("format", "Standard 2D") if image_metadata else "Standard"), body_style)],
        [Paragraph("DICOM Calibrated", body_style), Paragraph("Yes" if image_metadata and image_metadata.get("is_dicom") else "No (Standard 2D Image)", body_style)],
    ]
    if image_metadata and image_metadata.get("pixel_spacing_mm"):
        meta_data.append([
            Paragraph("Pixel Spacing", body_style),
            Paragraph(f"{image_metadata['pixel_spacing_mm']} mm", body_style)
        ])

    meta_table = Table(meta_data, colWidths=[150, 390])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Individual Classification Models
    story.append(Paragraph("2. Individual Deep Learning Classification Models", heading2_style))
    clf_data = [
        [
            Paragraph("<b>Model Architecture</b>", body_style),
            Paragraph("<b>Status</b>", body_style),
            Paragraph("<b>Predicted Class</b>", body_style),
            Paragraph("<b>Confidence</b>", body_style),
            Paragraph("<b>Stone Prob.</b>", body_style),
            Paragraph("<b>Non-Stone Prob.</b>", body_style),
            Paragraph("<b>Latency</b>", body_style),
        ]
    ]

    for arch, p in predictions.items():
        if p.get("available"):
            clf_data.append([
                Paragraph(arch.upper().replace("_", " "), body_style),
                Paragraph("Trained & Loaded", body_style),
                Paragraph(f"<b>{p.get('predicted_class', 'N/A')}</b>", body_style),
                Paragraph(f"{round(p.get('confidence', 0.0) * 100, 1)}%", body_style),
                Paragraph(f"{round(p.get('stone_probability', 0.0) * 100, 1)}%", body_style),
                Paragraph(f"{round(p.get('non_stone_probability', 0.0) * 100, 1)}%", body_style),
                Paragraph(f"{p.get('inference_time_ms', 0)} ms", body_style),
            ])
        else:
            clf_data.append([
                Paragraph(arch.upper().replace("_", " "), body_style),
                Paragraph("<font color='red'>Unavailable</font>", body_style),
                Paragraph("N/A", body_style),
                Paragraph("N/A", body_style),
                Paragraph("N/A", body_style),
                Paragraph("N/A", body_style),
                Paragraph("N/A", body_style),
            ])

    clf_table = Table(clf_data, colWidths=[110, 80, 80, 65, 70, 75, 60])
    clf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(clf_table)
    story.append(Spacer(1, 10))

    # Ensemble Output
    story.append(Paragraph("3. Multi-Model Ensemble Analysis", heading2_style))
    if ensemble and ensemble.get("available"):
        ens_pred = ensemble.get("predicted_class", "N/A")
        ens_conf = round(ensemble.get("confidence", 0.0) * 100, 1)
        ens_stone = round(ensemble.get("stone_probability", 0.0) * 100, 1)
        ens_non_stone = round(ensemble.get("non_stone_probability", 0.0) * 100, 1)
        contribs = ", ".join(ensemble.get("contributing_models", []))
        weights_info = str(ensemble.get("weights", {}))

        ens_data = [
            [Paragraph("<b>Ensemble Decision</b>", body_style), Paragraph(f"<b>{ens_pred}</b> ({ens_conf}% confidence)", body_style)],
            [Paragraph("Stone Probability", body_style), Paragraph(f"{ens_stone}%", body_style)],
            [Paragraph("Non-Stone Probability", body_style), Paragraph(f"{ens_non_stone}%", body_style)],
            [Paragraph("Contributing Classifiers", body_style), Paragraph(contribs, body_style)],
            [Paragraph("Normalized Model Weights", body_style), Paragraph(weights_info, body_style)],
            [Paragraph("Model Disagreement", body_style), Paragraph("Detected Disagreement" if ensemble.get("model_disagreement") else "Full Consensus", body_style)],
        ]
        ens_table = Table(ens_data, colWidths=[150, 390])
        ens_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FEFCBF")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ens_table)
    else:
        story.append(Paragraph("Ensemble prediction unavailable (insufficient trained classification models).", body_style))
    story.append(Spacer(1, 10))

    # YOLO Detection
    story.append(Paragraph("4. YOLOv8 Object Detection (Localization)", heading2_style))
    if detection and detection.get("available"):
        det_summary = f"Detected {detection.get('num_detections', 0)} region(s) using YOLOv8."
        story.append(Paragraph(det_summary, body_style))
    else:
        det_msg = "Detection model not trained or unavailable. Stone localization bounding-box annotations required."
        story.append(Paragraph(det_msg, body_style))
    story.append(Spacer(1, 10))

    # Measurements & Calibrations
    story.append(Paragraph("5. Physical Measurements & Calibrations", heading2_style))
    if measurement_info:
        story.append(Paragraph(f"<b>Status:</b> {measurement_info.get('measurement_status', 'N/A')}", body_style))
        for lim in measurement_info.get("limitations", []):
            story.append(Paragraph(f"• {lim}", body_style))
    else:
        story.append(Paragraph("No physical measurement calibrated for this examination.", body_style))
    story.append(Spacer(1, 10))

    # Mandatory Scientific & Medical Disclaimer
    story.append(Paragraph(
        "<b>MANDATORY MEDICAL & RESEARCH DISCLAIMER:</b><br/>"
        "This report was generated by an academic research artificial intelligence system. "
        "The classification outputs, probability scores, and ensemble results are algorithmic estimates "
        "and do NOT constitute a medical diagnosis, clinical radiologist verification, or treatment recommendation. "
        "Model confidence must not be interpreted as an absolute probability of clinical disease. "
        "Clinical diagnosis requires thorough review by a qualified radiologist or physician.",
        disclaimer_style,
    ))

    doc.build(story)
    return pdf_path
