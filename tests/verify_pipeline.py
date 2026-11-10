"""
End-to-End Pipeline Verification Script
Tests loading a real CT scan image from the dataset, runs inference across
all 3 trained models (MobileNetV2, ResNet50, EfficientNet-B0), synthesizes
ensemble consensus, computes Grad-CAM heatmaps, stores the study in SQLite,
and compiles a clinical PDF research report.
"""
from PIL import Image
from pathlib import Path
from backend.services.classification import run_all_available_classifiers
from backend.services.ensemble import compute_ensemble_prediction
from backend.services.explainability import generate_gradcam_explanation
from backend.database.db_manager import save_examination, get_all_examinations
from backend.reporting.pdf_generator import generate_pdf_report

def main():
    print("=" * 60)
    print("Executing End-to-End Deep Learning Verification Pipeline...")
    print("=" * 60)

    # 1. Load real test slice
    sample_path = Path("dataset/Kindey Stone Dataset/Kindey Stone Dataset/Original/Stone/P037_FA_M_S_I01.jpg")
    if not sample_path.exists():
        candidates = list(Path("dataset").rglob("*.jpg"))
        if not candidates:
            raise FileNotFoundError("No sample JPG images found in dataset folder.")
        sample_path = candidates[0]

    image = Image.open(sample_path).convert("RGB")
    print(f"\n[1] Loaded Real Clinical Scan: {sample_path.name}")
    print(f"    Dimensions: {image.width} x {image.height} pixels | Format: RGB")

    # 2. Run All 3 Classifiers Independently
    print("\n[2] Executing Independent Inferences:")
    preds = run_all_available_classifiers(image)
    for arch, p in preds.items():
        if p.get("available"):
            cls_name = p["predicted_class"]
            conf = p["confidence"] * 100
            stone_p = p["stone_probability"] * 100
            ns_p = p["non_stone_probability"] * 100
            lat = p["inference_time_ms"]
            print(f"    • {arch.upper():16}: Prediction={cls_name:9} | Conf={conf:5.1f}% | Stone={stone_p:5.1f}% | NonStone={ns_p:5.1f}% | Latency={lat:5.1f}ms")
        else:
            print(f"    • {arch.upper():16}: UNAVAILABLE")

    # 3. Multi-Model Ensemble
    print("\n[3] Computing Weighted Ensemble Consensus:")
    ens = compute_ensemble_prediction(preds)
    if ens.get("available"):
        e_cls = ens["predicted_class"]
        e_conf = ens["confidence"] * 100
        e_stone = ens["stone_probability"] * 100
        e_ns = ens["non_stone_probability"] * 100
        disag = "Disagreement Flagged" if ens.get("model_disagreement") else "Full Agreement"
        print(f"    • Ensemble Decision : {e_cls} ({e_conf:.1f}% confidence)")
        print(f"    • Class Breakdown   : Stone={e_stone:.1f}% | Non-Stone={e_ns:.1f}%")
        print(f"    • Consensus Status  : {disag} ({ens.get('consensus_percentage', 100)}% consensus)")
        print(f"    • Weights Used      : {ens.get('weights')}")

    # 4. Grad-CAM Explainability on ResNet50 & EfficientNet-B0
    print("\n[4] Generating Explainable AI (Grad-CAM) Visualizations:")
    for test_arch in ["resnet50", "mobilenet_v2", "efficientnet_b0"]:
        cam = generate_gradcam_explanation(image, test_arch, target_class_name="Stone")
        print(f"    • {test_arch.upper():16}: Grad-CAM Generated = {cam.get('success')} (Target={cam.get('target_class')})")

    # 5. Database Archiving
    print("\n[5] Archiving Study in SQLite Research Database:")
    exam_id = "VERIFY_2026_01"
    save_examination(
        exam_id=exam_id,
        predictions=preds,
        ensemble=ens,
        filename=sample_path.name,
        notes="Automated verification test case on genuine clinical stone slice.",
    )
    all_exams = get_all_examinations(limit=5)
    print(f"    • Database Record Saved: Exam #{exam_id} (Total stored in DB: {len(all_exams)})")

    # 6. PDF Report Generation
    print("\n[6] Compiling Downloadable ReportLab PDF Documentation:")
    pdf_path = generate_pdf_report(
        exam_id=exam_id,
        predictions=preds,
        ensemble=ens,
        image_metadata={"format": "JPEG", "is_dicom": False},
    )
    print(f"    • PDF Generated : {pdf_path.name} ({pdf_path.stat().st_size} bytes)")
    print("\n" + "=" * 60)
    print("End-to-End Verification Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
