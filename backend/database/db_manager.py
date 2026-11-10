"""
Database Management Service using SQLite
Maintains persistent research examination history, classifier outputs,
ensemble conclusions, and patient study records.
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from config.settings import DATABASE_PATH, DATABASE_DIR


def get_db_connection() -> sqlite3.Connection:
    """Create and return a connection to the SQLite database."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS examinations (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            filename TEXT,
            image_format TEXT,
            is_dicom INTEGER DEFAULT 0,
            pixel_spacing_mm TEXT,
            predictions_json TEXT NOT NULL,
            ensemble_json TEXT,
            detection_json TEXT,
            contributing_models TEXT,
            predicted_class TEXT,
            confidence REAL,
            measurement_status TEXT,
            warnings_json TEXT,
            notes TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            model_name TEXT PRIMARY KEY,
            architecture TEXT NOT NULL,
            checkpoint_path TEXT NOT NULL,
            class_mapping_json TEXT NOT NULL,
            best_val_acc REAL,
            best_val_loss REAL,
            epochs_trained INTEGER,
            training_timestamp TEXT,
            status TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_examination(
    exam_id: str,
    predictions: Dict[str, Any],
    ensemble: Optional[Dict[str, Any]] = None,
    detection: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    image_format: Optional[str] = None,
    is_dicom: bool = False,
    pixel_spacing_mm: Optional[List[float]] = None,
    measurement_status: Optional[str] = None,
    warnings: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> None:
    """Save an analysis record into the database."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    pred_class = None
    conf = None
    contributing = []

    if ensemble and "predicted_class" in ensemble:
        pred_class = ensemble["predicted_class"]
        conf = ensemble.get("confidence", 0.0)
        contributing = ensemble.get("contributing_models", [])
    elif predictions:
        # Fallback to the first available model
        for m_name, p in predictions.items():
            if p.get("available") and "predicted_class" in p:
                pred_class = p["predicted_class"]
                conf = p.get("confidence", 0.0)
                contributing = [m_name]
                break

    cursor.execute("""
        INSERT OR REPLACE INTO examinations (
            id, timestamp, filename, image_format, is_dicom,
            pixel_spacing_mm, predictions_json, ensemble_json, detection_json,
            contributing_models, predicted_class, confidence,
            measurement_status, warnings_json, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        exam_id,
        datetime.now().isoformat(),
        filename or "uploaded_image",
        image_format or "standard",
        1 if is_dicom else 0,
        json.dumps(pixel_spacing_mm) if pixel_spacing_mm else None,
        json.dumps(predictions),
        json.dumps(ensemble) if ensemble else None,
        json.dumps(detection) if detection else None,
        json.dumps(contributing),
        pred_class,
        conf,
        measurement_status or "None",
        json.dumps(warnings or []),
        notes or ""
    ))

    conn.commit()
    conn.close()


def get_all_examinations(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve recent examinations from database."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM examinations
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["predictions"] = json.loads(d["predictions_json"]) if d["predictions_json"] else {}
        d["ensemble"] = json.loads(d["ensemble_json"]) if d["ensemble_json"] else None
        d["detection"] = json.loads(d["detection_json"]) if d["detection_json"] else None
        d["warnings"] = json.loads(d["warnings_json"]) if d["warnings_json"] else []
        d["contributing_models"] = json.loads(d["contributing_models"]) if d["contributing_models"] else []
        results.append(d)

    conn.close()
    return results


def get_examination_by_id(exam_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve single examination record by ID."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM examinations WHERE id = ?", (exam_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    d = dict(row)
    d["predictions"] = json.loads(d["predictions_json"]) if d["predictions_json"] else {}
    d["ensemble"] = json.loads(d["ensemble_json"]) if d["ensemble_json"] else None
    d["detection"] = json.loads(d["detection_json"]) if d["detection_json"] else None
    d["warnings"] = json.loads(d["warnings_json"]) if d["warnings_json"] else []
    d["contributing_models"] = json.loads(d["contributing_models"]) if d["contributing_models"] else []
    return d


# Initialize tables upon module import
init_db()
