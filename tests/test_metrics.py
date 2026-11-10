"""
Unit tests for evaluation metrics and confusion matrix calculations.
"""
import pytest
from evaluation.metrics import compute_metrics


def test_perfect_classification_metrics():
    """Verify metrics for perfect classification output."""
    # 0 = Non-Stone, 1 = Stone
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = [0.1, 0.05, 0.95, 0.99]

    res = compute_metrics(y_true, y_pred, y_prob)

    assert res["accuracy"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall_sensitivity"] == 1.0
    assert res["specificity"] == 1.0
    assert res["f1_score"] == 1.0
    assert res["roc_auc"] == 1.0
    assert res["confusion_matrix"]["tn"] == 2
    assert res["confusion_matrix"]["tp"] == 2
    assert res["confusion_matrix"]["fp"] == 0
    assert res["confusion_matrix"]["fn"] == 0


def test_mixed_classification_metrics():
    """Verify metrics calculation on realistic mixed predictions."""
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 0, 1]  # 1 FN, 1 FP

    res = compute_metrics(y_true, y_pred)
    # TP=2, FN=1, TN=1, FP=1
    assert res["confusion_matrix"]["tp"] == 2
    assert res["confusion_matrix"]["fn"] == 1
    assert res["confusion_matrix"]["tn"] == 1
    assert res["confusion_matrix"]["fp"] == 1

    # Sensitivity = TP/(TP+FN) = 2/3 = 0.6667
    assert abs(res["recall_sensitivity"] - 0.6667) < 1e-3
    # Specificity = TN/(TN+FP) = 1/2 = 0.5000
    assert abs(res["specificity"] - 0.5000) < 1e-3


def test_zero_division_safety():
    """Verify zero division handled gracefully without unhandled exceptions."""
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]  # No positives at all
    res = compute_metrics(y_true, y_pred)
    assert res["recall_sensitivity"] == 0.0
    assert res["precision"] == 0.0
    assert res["specificity"] == 1.0
