# AI-Based Kidney Stone Detection & Multi-Model Classification System

A production-grade, academic research application for detecting and classifying kidney stones in CT imaging using multiple independent deep learning architectures (**MobileNetV2**, **ResNet50**, **EfficientNet-B0**), combined ensemble synthesis, **Grad-CAM** interpretability, and **YOLOv8** stone localization.

---

## 🔬 System Overview & Scientific Design

- **Canonical Class Mapping**:
  - `0 = Non-Stone`
  - `1 = Stone`
  - Stored strictly in checkpoint metadata and maintained uniformly across all data loaders, model heads, metrics, and report generators.
- **Zero-Leakage Splitting**:
  - Identifies embedded patient IDs (`P001`, `P037`, etc.) and root parent image IDs from filenames (e.g. `P001_FA_M_NS_I01`).
  - Isolates all parent slices and synthetic augmentations (`_CR`, `_DO`, `_GB`, `_FH`, `_FV`) of each patient strictly into a single partition.
  - Zero patient overlap between `train`, `validation`, and `test` partitions.
  - Validation and held-out test partitions evaluate purely on real clinical original slices.
- **Independent Deep Learning Classifiers**:
  - **MobileNetV2**: Lightweight, low-latency convolutional architecture.
  - **ResNet50**: Deep residual network with residual bottleneck blocks.
  - **EfficientNet-B0**: Compound-scaled neural network balancing accuracy and FLOPS.
- **Ensemble Synthesis**:
  - Weighted probability aggregation over available trained checkpoints with model consensus and disagreement quantification.
- **YOLOv8 Stone Localization**:
  - Dedicated object detection service that triggers only when valid bounding-box annotations exist.
  - Does not invent bounding boxes or confuse whole-image classification with spatial localization.
- **Grad-CAM Visual Explainability**:
  - Real gradient-weighted class activation mapping highlighting salient convolutional feature activations for each architecture.
- **DICOM & Physical Measurements**:
  - Reads `PixelSpacing` and `SliceThickness` metadata via `pydicom`.
  - Calculates physical millimeter dimensions for calibrated scans or explicitly flags uncalibrated 2D pixel areas.
- **Persistent Database & PDF Reports**:
  - SQLite storage for research examination history.
  - Formatted ReportLab clinical research reports with disclaimers.

---

## 📁 Repository Structure

```
kidney stone detection/
├── streamlit_app.py               # Main Application & Navigation Hub
├── requirements.txt               # Project Dependencies
├── README.md                      # Documentation
├── .env.example                   # Environment Configuration
├── config/
│   ├── __init__.py
│   └── settings.py                # Central Canonical Configuration & Device Detection
├── backend/
│   ├── database/
│   │   ├── __init__.py
│   │   └── db_manager.py          # SQLite Research History Manager
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── pdf_generator.py       # ReportLab PDF Generator
│   └── services/
│       ├── __init__.py
│       ├── image_loader.py        # Safe Standard & DICOM Loader
│       ├── preprocessing.py       # Preprocessing & PyTorch Dataset
│       ├── classification.py      # Independent Model Builders & Inference
│       ├── ensemble.py            # Normalized Weighted Ensemble
│       ├── detection.py           # YOLOv8 Detection Service
│       ├── measurement.py         # Physical Calibration & Millimeter Measurement
│       ├── explainability.py      # Real Grad-CAM Implementation
│       └── ui_theme.py            # Glassmorphism Modern UI Theme
├── training/
│   ├── __init__.py
│   ├── inspect_dataset.py         # Dataset Discovery & Casing/Spelling Validator
│   ├── split_dataset.py           # Patient-Stratified Zero-Leakage Splitting
│   ├── train_classifier.py        # Sequential PyTorch Classifier Trainer
│   ├── train_yolo.py              # YOLOv8 Training & YAML Validator
│   └── evaluate_models.py         # Held-Out Test Set Benchmarking
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # Accuracy, Specificity, Sensitivity, ROC-AUC
│   └── eval_results.json          # Saved Test Evaluation Benchmarks
├── pages/
│   ├── 1_Dashboard.py             # Telemetry & Status Dashboard
│   ├── 2_Dataset_Manager.py       # Dataset Inspection & Splitting Controls
│   ├── 3_Training_Center.py       # Interactive Model Training Controls
│   ├── 4_CT_Image_Analysis.py     # Multi-Model Inference & Report Export
│   ├── 5_Model_Comparison.py      # Held-Out Metrics & Confusion Matrices
│   ├── 6_Training_History.py      # Epoch Loss & Accuracy Curves
│   ├── 7_Explainable_AI.py        # Grad-CAM Heatmaps & Overlays
│   ├── 8_YOLO_Detection.py        # Bounding-Box Detection & Localization
│   ├── 9_Model_Management.py      # Checkpoint Registry & Integrity Verifier
│   ├── 10_Patient_History.py      # SQLite Examination Case Audit
│   └── 11_Reports.py              # PDF Report Management
├── data/
│   └── splits/                    # train.csv, val.csv, test.csv, manifest.json
├── models/
│   ├── classification/            # Saved .pth Checkpoints and History JSONs
│   └── detection/                 # YOLOv8 best.pt Checkpoint
├── reports/                       # Generated Clinical PDF Reports
└── tests/
    ├── test_dataset.py            # Mapping and Leakage Tests
    ├── test_image_loader.py       # Image Loading and Corruption Tests
    ├── test_models.py             # Shapes, Checkpoints, and Mismatch Rejection
    ├── test_ensemble.py           # Weighting and Disagreement Tests
    ├── test_metrics.py            # Confusion Matrix and Clinical Metrics
    └── test_detection_and_reporting.py # YOLO Fallback, DB, and PDF Tests
```

---

## 🚀 Getting Started

### 1. Launch the Streamlit Research Application
Run the following command in PowerShell:
```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

### 2. Run the Automated Unit Test Suite
To verify all 20 test specifications across dataset integrity, leakage isolation, checkpoint handling, and metrics:
```powershell
.\venv\Scripts\python.exe -m pytest tests -v
```

### 3. Dataset Splitting & Training CLI
To inspect the dataset structure:
```powershell
.\venv\Scripts\python.exe -m training.inspect_dataset
```

To regenerate patient-isolated splits:
```powershell
.\venv\Scripts\python.exe -m training.split_dataset
```

To run model training via CLI (or use the interactive **Training Center** in Streamlit):
```powershell
.\venv\Scripts\python.exe -c "from training.train_classifier import train_single_classifier; train_single_classifier('mobilenet_v2', epochs=5, batch_size=32)"
```
