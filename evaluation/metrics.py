"""
Evaluation Metrics Computation Engine
Calculates Accuracy, Sensitivity/Recall, Specificity, Precision, F1-Score,
ROC-AUC, and Confusion Matrix with zero-division safety.
"""
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from config.settings import CLASS_NAMES, CLASS_TO_IDX


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob_stone: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive clinical evaluation metrics from held-out predictions.
    Canonical index: 0 = Non-Stone (Negative), 1 = Stone (Positive).
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    if len(y_t) == 0:
        return {"error": "Empty evaluation vectors provided"}

    acc = float(accuracy_score(y_t, y_p))
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])

    # Canonical mapping: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity / Recall (True Positive Rate)
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Specificity (True Negative Rate)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Precision (Positive Predictive Value)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # F1 Score
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    # ROC-AUC
    roc_auc = None
    if y_prob_stone is not None and len(np.unique(y_t)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_t, y_prob_stone))
        except Exception:
            roc_auc = None

    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "sample_count": int(len(y_t)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
            "labels": CLASS_NAMES,
        },
    }
