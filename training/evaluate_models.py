"""
Model Evaluation Engine
Evaluates trained models strictly on the held-out test split,
computes inference benchmarks, evaluates ensemble performance, and saves results.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from config.settings import (
    SPLITS_DIR,
    BASE_DIR,
    CLASS_NAMES,
    get_device,
)
from backend.services.preprocessing import KidneyStoneDataset, get_transforms
from backend.services.classification import (
    load_checkpoint,
    is_model_available,
    get_checkpoint_path,
    SUPPORTED_ARCHITECTURES,
)
from backend.services.ensemble import compute_ensemble_prediction
from evaluation.metrics import compute_metrics

EVAL_RESULTS_PATH = BASE_DIR / "evaluation" / "eval_results.json"


def evaluate_single_model(
    architecture: str,
    test_csv_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single trained classifier on the held-out test set.
    """
    arch = architecture.lower()
    if not is_model_available(arch):
        return {
            "architecture": arch,
            "evaluated": False,
            "status": "Checkpoint not available. Train model first.",
        }

    if device is None:
        device = get_device()

    csv_path = test_csv_path or (SPLITS_DIR / "test.csv")
    if not csv_path.exists():
        return {
            "architecture": arch,
            "evaluated": False,
            "status": f"Test split CSV not found at {csv_path}",
        }

    model, checkpoint = load_checkpoint(arch, device=device)
    model.eval()

    dataset = KidneyStoneDataset(
        csv_file_or_df=csv_path,
        transform=get_transforms(split="test", apply_augmentation=False),
        max_samples=max_samples,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    y_true = []
    y_pred = []
    y_prob_stone = []

    start_eval_time = time.perf_counter()

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(targets.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob_stone.extend(probs[:, 1].tolist())

    total_eval_time = time.perf_counter() - start_eval_time
    avg_latency_ms = round((total_eval_time / max(1, len(dataset))) * 1000.0, 2)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob_stone)

    # File size in MB
    cp_path = get_checkpoint_path(arch)
    model_size_mb = round(cp_path.stat().st_size / (1024**2), 2) if cp_path.exists() else 0.0

    return {
        "architecture": arch,
        "evaluated": True,
        "metrics": metrics,
        "avg_latency_ms": avg_latency_ms,
        "model_size_mb": model_size_mb,
        "test_samples": len(dataset),
        "split_used": str(csv_path.name),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob_stone": y_prob_stone,
    }


def evaluate_all_available_models(
    test_csv_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate all trained classifiers and compute held-out ensemble metrics.
    Saves results to evaluation/eval_results.json.
    """
    results: Dict[str, Any] = {
        "models": {},
        "ensemble": None,
        "summary": [],
    }

    evaluated_models = {}

    for arch in SUPPORTED_ARCHITECTURES:
        res = evaluate_single_model(
            architecture=arch,
            test_csv_path=test_csv_path,
            device=device,
            max_samples=max_samples,
        )
        results["models"][arch] = res
        if res.get("evaluated"):
            evaluated_models[arch] = res

    # Evaluate Ensemble if 2 or more models are evaluated
    if len(evaluated_models) >= 2:
        first_m = list(evaluated_models.values())[0]
        y_true = first_m["y_true"]
        n_samples = len(y_true)

        ensemble_probs_stone = []
        ensemble_preds = []

        for i in range(n_samples):
            pred_dict = {}
            for arch, m_res in evaluated_models.items():
                p_stone = m_res["y_prob_stone"][i]
                pred_dict[arch] = {
                    "available": True,
                    "probabilities": {
                        "Non-Stone": 1.0 - p_stone,
                        "Stone": p_stone,
                    },
                }
            ens_res = compute_ensemble_prediction(pred_dict)
            p_stone_ens = ens_res["probabilities"]["Stone"]
            ensemble_probs_stone.append(p_stone_ens)
            ensemble_preds.append(1 if p_stone_ens >= 0.5 else 0)

        ens_metrics = compute_metrics(y_true, ensemble_preds, ensemble_probs_stone)
        results["ensemble"] = {
            "evaluated": True,
            "architecture": "Ensemble (Averaged)",
            "contributing_models": list(evaluated_models.keys()),
            "metrics": ens_metrics,
            "test_samples": n_samples,
        }

    # Save to disk (excluding large raw arrays for concise storage)
    save_copy = {
        "models": {},
        "ensemble": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for arch, m in results["models"].items():
        if m.get("evaluated"):
            save_copy["models"][arch] = {k: v for k, v in m.items() if k not in ["y_true", "y_pred", "y_prob_stone"]}
        else:
            save_copy["models"][arch] = m

    if results["ensemble"]:
        save_copy["ensemble"] = results["ensemble"]

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(save_copy, f, indent=2)

    return results


def load_evaluation_results() -> Optional[Dict[str, Any]]:
    """Load existing evaluation results from disk."""
    if EVAL_RESULTS_PATH.exists():
        try:
            with open(EVAL_RESULTS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Evaluating all trained classifiers on the held-out test set...")
    print("=" * 60)
    res = evaluate_all_available_models()
    print("\n[EVALUATION COMPLETE] Summary of Test Set Metrics (381 Patients/Slices):")
    for arch, m in res["models"].items():
        if m.get("evaluated"):
            met = m["metrics"]
            acc = met["accuracy"] * 100
            sens = met["recall_sensitivity"] * 100
            spec = met["specificity"] * 100
            f1 = met["f1_score"]
            lat = m["avg_latency_ms"]
            size = m["model_size_mb"]
            print(f"- {arch.upper():16}: Acc={acc:5.1f}% | Sens={sens:5.1f}% | Spec={spec:5.1f}% | F1={f1:.3f} | Latency={lat:5.1f}ms | Size={size:5.1f}MB")

    if res.get("ensemble") and res["ensemble"].get("evaluated"):
        emet = res["ensemble"]["metrics"]
        e_acc = emet["accuracy"] * 100
        e_sens = emet["recall_sensitivity"] * 100
        e_spec = emet["specificity"] * 100
        e_f1 = emet["f1_score"]
        print(f"- {'ENSEMBLE':16}: Acc={e_acc:5.1f}% | Sens={e_sens:5.1f}% | Spec={e_spec:5.1f}% | F1={e_f1:.3f}")
    print("=" * 60 + "\n")
