"""
Unit tests for multi-model ensemble aggregation and weighting logic.
"""
import pytest
from backend.services.ensemble import compute_ensemble_prediction


def test_ensemble_all_unavailable():
    """Verify clean status when no models are available."""
    preds = {
        "mobilenet_v2": {"available": False},
        "resnet50": {"available": False},
    }
    ens = compute_ensemble_prediction(preds)
    assert ens["available"] is False
    assert len(ens["contributing_models"]) == 0


def test_ensemble_single_model():
    """Verify single model behaves transparently without pretending multiple models contributed."""
    preds = {
        "mobilenet_v2": {
            "available": True,
            "probabilities": {"Non-Stone": 0.20, "Stone": 0.80},
        },
        "resnet50": {"available": False},
    }
    ens = compute_ensemble_prediction(preds)
    assert ens["available"] is True
    assert ens["is_single_model"] is True
    assert ens["contributing_models"] == ["mobilenet_v2"]
    assert ens["predicted_class"] == "Stone"
    assert ens["stone_probability"] == 0.80
    assert ens["non_stone_probability"] == 0.20


def test_ensemble_multi_model_equal_weights():
    """Verify equal weight averaging over multiple available models."""
    preds = {
        "mobilenet_v2": {
            "available": True,
            "probabilities": {"Non-Stone": 0.10, "Stone": 0.90},
        },
        "resnet50": {
            "available": True,
            "probabilities": {"Non-Stone": 0.30, "Stone": 0.70},
        },
    }
    ens = compute_ensemble_prediction(preds)
    assert ens["available"] is True
    assert ens["is_single_model"] is False
    assert len(ens["contributing_models"]) == 2
    # Mean of [0.90, 0.70] is 0.80
    assert abs(ens["stone_probability"] - 0.80) < 1e-4
    assert abs(ens["non_stone_probability"] - 0.20) < 1e-4
    assert ens["predicted_class"] == "Stone"
    assert ens["model_disagreement"] is False


def test_ensemble_disagreement_detection():
    """Verify disagreement is flagged when models predict opposing classes."""
    preds = {
        "mobilenet_v2": {
            "available": True,
            "probabilities": {"Non-Stone": 0.10, "Stone": 0.90},  # Predicts Stone
        },
        "resnet50": {
            "available": True,
            "probabilities": {"Non-Stone": 0.75, "Stone": 0.25},  # Predicts Non-Stone
        },
    }
    ens = compute_ensemble_prediction(preds)
    assert ens["model_disagreement"] is True
    assert ens["consensus_percentage"] == 50.0
