"""
Ensemble Prediction Service
Combines probability distributions from independently trained classification models
using normalized weighted averaging with disagreement analysis and scientific disclaimers.
"""
from typing import Dict, Any, List, Optional
import numpy as np

from config.settings import CLASS_NAMES


def compute_ensemble_prediction(
    model_predictions: Dict[str, Dict[str, Any]],
    custom_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute ensemble probability using weighted averaging over available models.
    """
    contributing_models = []
    prob_vectors = []
    raw_weights = []

    # Filter only available models with valid probabilities
    for arch, pred in model_predictions.items():
        if pred.get("available") and "probabilities" in pred:
            probs = pred["probabilities"]
            if "Non-Stone" in probs and "Stone" in probs:
                contributing_models.append(arch)
                # Canonical order: [Non-Stone, Stone]
                prob_vectors.append([probs["Non-Stone"], probs["Stone"]])

                weight = 1.0
                if custom_weights and arch in custom_weights:
                    weight = max(0.0, float(custom_weights[arch]))
                raw_weights.append(weight)

    if not contributing_models:
        return {
            "available": False,
            "status": "No trained classification models available to compute ensemble.",
            "contributing_models": [],
            "disclaimer": "Research prototype only. Not clinically validated.",
        }

    # Normalize weights
    total_w = sum(raw_weights)
    if total_w <= 0:
        normalized_weights = [1.0 / len(raw_weights)] * len(raw_weights)
    else:
        normalized_weights = [w / total_w for w in raw_weights]

    weights_dict = {
        arch: round(w, 4) for arch, w in zip(contributing_models, normalized_weights)
    }

    # Weighted average probabilities
    prob_matrix = np.array(prob_vectors)  # shape: (N, 2)
    weights_col = np.array(normalized_weights).reshape(-1, 1)  # shape: (N, 1)
    ensemble_probs = np.sum(prob_matrix * weights_col, axis=0)  # shape: (2,)

    non_stone_prob = float(ensemble_probs[0])
    stone_prob = float(ensemble_probs[1])

    if stone_prob >= non_stone_prob:
        predicted_class = "Stone"
        confidence = stone_prob
    else:
        predicted_class = "Non-Stone"
        confidence = non_stone_prob

    # Model agreement/disagreement calculation
    individual_preds = [
        "Stone" if p[1] >= p[0] else "Non-Stone" for p in prob_vectors
    ]
    stone_votes = individual_preds.count("Stone")
    non_stone_votes = individual_preds.count("Non-Stone")

    disagreement = False
    consensus_pct = 100.0
    if len(contributing_models) > 1:
        max_votes = max(stone_votes, non_stone_votes)
        consensus_pct = round((max_votes / len(contributing_models)) * 100.0, 1)
        disagreement = stone_votes > 0 and non_stone_votes > 0

    is_single_model = len(contributing_models) == 1

    return {
        "available": True,
        "is_single_model": is_single_model,
        "contributing_models": contributing_models,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "stone_probability": round(stone_prob, 4),
        "non_stone_probability": round(non_stone_prob, 4),
        "probabilities": {
            "Non-Stone": round(non_stone_prob, 4),
            "Stone": round(stone_prob, 4),
        },
        "weights": weights_dict,
        "weights_type": "Default Equal Weights" if not custom_weights else "Custom Normalized Weights",
        "model_disagreement": disagreement,
        "consensus_percentage": consensus_pct,
        "vote_counts": {
            "Stone": stone_votes,
            "Non-Stone": non_stone_votes,
        },
        "disclaimer": (
            "RESEARCH USE ONLY: The ensemble combination represents a mathematical aggregation "
            "of deep learning model confidence and is NOT a calibrated clinical diagnosis. "
            "Model predictions should never be used as the sole basis for clinical treatment."
        ),
    }
