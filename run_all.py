"""
========================================================================
  RUN ALL — Kidney Stone Detection Full Pipeline
  Steps:
    1. Generate synthetic CT demo dataset  (600 images)
    2. Create train/val/test splits        (70/15/15)
    3. Train MobileNetV2, ResNet50, EfficientNet-B0
    4. Print final summary
========================================================================
"""
import sys, time, shutil, random, json
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATASET_DIR   = ROOT / "dataset" / "Original"
SPLITS_DIR    = ROOT / "data" / "splits"
MODELS_DIR    = ROOT / "models" / "classification"

STONE_COUNT     = 300
NON_STONE_COUNT = 300
IMG_SIZE        = 224

EPOCHS        = 15
BATCH_SIZE    = 16
LR            = 1e-4
PATIENCE      = 5

SEP  = "=" * 70
DASH = "─" * 70

# ─────────────────────────────────────────────────────────
# STEP 1 — Dataset generation
# ─────────────────────────────────────────────────────────

def _ct_background(size=224):
    rng = np.random.default_rng()
    base = rng.integers(80, 130, size=(size, size), dtype=np.uint8)
    img = Image.fromarray(base, mode="L")
    return img.filter(ImageFilter.GaussianBlur(radius=2))

def _stone_image(size=224):
    img = _ct_background(size)
    draw = ImageDraw.Draw(img)
    cx = random.randint(size//4, 3*size//4)
    cy = random.randint(size//4, 3*size//4)
    rx = random.randint(5, 20); ry = random.randint(4, 15)
    bright = random.randint(200, 255)
    draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=bright)
    if random.random() > 0.5:
        draw.ellipse([cx-rx-3, cy-ry-3, cx+rx+3, cy+ry+3], outline=180)
    kx, ky = size//2, size//2
    krx = random.randint(50,80); kry = random.randint(60,90)
    draw.ellipse([kx-krx, ky-kry, kx+krx, ky+kry], outline=50)
    return img.filter(ImageFilter.GaussianBlur(radius=0.8)).convert("RGB")

def _nonstone_image(size=224):
    img = _ct_background(size)
    draw = ImageDraw.Draw(img)
    kx, ky = size//2, size//2
    krx = random.randint(50,80); kry = random.randint(60,90)
    draw.ellipse([kx-krx, ky-kry, kx+krx, ky+kry], outline=50)
    for _ in range(random.randint(2,5)):
        x = random.randint(kx-krx+10, kx+krx-10)
        y = random.randint(ky-kry+10, ky+kry-10)
        r = random.randint(5,15)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=random.randint(100,150))
    return img.filter(ImageFilter.GaussianBlur(radius=1.0)).convert("RGB")

def step1_generate_dataset():
    print(f"\n{DASH}")
    print("  STEP 1 — Generating Synthetic CT Demo Dataset")
    print(DASH)

    # Clean and recreate
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    classes = {"Stone": (STONE_COUNT, _stone_image),
               "Non-Stone": (NON_STONE_COUNT, _nonstone_image)}

    for cls_name, (count, fn) in classes.items():
        cls_dir = DATASET_DIR / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, count + 1):
            img = fn()
            fname = f"P{i:03d}_{cls_name.replace('-','_')}_CT_{i:04d}.jpg"
            img.save(cls_dir / fname, "JPEG", quality=90)
            if i % 100 == 0:
                print(f"    {cls_name}: {i}/{count}")

    total = STONE_COUNT + NON_STONE_COUNT
    print(f"\n  ✅ Dataset ready — {total} images "
          f"(Stone: {STONE_COUNT}, Non-Stone: {NON_STONE_COUNT})")

# ─────────────────────────────────────────────────────────
# STEP 2 — Splits
# ─────────────────────────────────────────────────────────

def step2_split_dataset():
    print(f"\n{DASH}")
    print("  STEP 2 — Creating Train / Val / Test Splits  (70 / 15 / 15)")
    print(DASH)

    # Clean old splits
    if SPLITS_DIR.exists():
        shutil.rmtree(SPLITS_DIR)

    from training.split_dataset import create_safe_dataset_split
    manifest = create_safe_dataset_split(dataset_root=DATASET_DIR)

    sc = manifest["split_counts"]
    lv = manifest["leakage_verification"]
    print(f"  Train : {sc['train']['total']:>4} images  "
          f"(Stone {sc['train']['by_class']['Stone']}, "
          f"Non-Stone {sc['train']['by_class']['Non-Stone']})")
    print(f"  Val   : {sc['val']['total']:>4} images  "
          f"(Stone {sc['val']['by_class']['Stone']}, "
          f"Non-Stone {sc['val']['by_class']['Non-Stone']})")
    print(f"  Test  : {sc['test']['total']:>4} images  "
          f"(Stone {sc['test']['by_class']['Stone']}, "
          f"Non-Stone {sc['test']['by_class']['Non-Stone']})")
    leak = "NONE ✅" if lv["leakage_free"] else "⚠️  DETECTED"
    print(f"  Leakage : {leak}")

# ─────────────────────────────────────────────────────────
# STEP 3 — Train models
# ─────────────────────────────────────────────────────────

def progress_callback(info):
    mark = "✅" if info["improved"] else "  "
    print(f"    {mark} Epoch {info['epoch']:>2}/{info['total_epochs']} | "
          f"Train {info['train_acc']:5.1f}% loss {info['train_loss']:.4f} | "
          f"Val {info['val_acc']:5.1f}% loss {info['val_loss']:.4f} | "
          f"Best {info['best_val_acc']:5.1f}%  p={info['patience_counter']}/{PATIENCE}")
    sys.stdout.flush()

def step3_train_models():
    from training.train_classifier import train_single_classifier
    from backend.services.classification import SUPPORTED_ARCHITECTURES

    print(f"\n{DASH}")
    print("  STEP 3 — Training Classification Models")
    print(f"  Architectures: {', '.join(SUPPORTED_ARCHITECTURES)}")
    print(f"  Epochs {EPOCHS} | Batch {BATCH_SIZE} | LR {LR} | Patience {PATIENCE}")
    print(DASH)

    # Clean old checkpoints
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.pth"):
            f.unlink()
        for f in MODELS_DIR.glob("*_history.json"):
            f.unlink()

    results = {}
    overall_t0 = time.time()

    for arch in SUPPORTED_ARCHITECTURES:
        print(f"\n  🧠 {arch.upper().replace('_',' ')}")
        t0 = time.time()
        res = train_single_classifier(
            architecture=arch,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            patience=PATIENCE,
            progress_callback=progress_callback,
        )
        elapsed = time.time() - t0
        results[arch] = res
        print(f"\n  ✔ {arch} done in {elapsed:.0f}s | "
              f"Best val acc: {res['best_val_acc']}% | "
              f"Best val loss: {res['best_val_loss']}")

    return results, time.time() - overall_t0

# ─────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────

def print_summary(results, total_secs):
    print(f"\n{SEP}")
    print("  🎉  TRAINING COMPLETE")
    print(SEP)
    for arch, res in results.items():
        ckpt = Path(res["checkpoint_path"]).name
        print(f"  {arch:<20}  val_acc={res['best_val_acc']:>6}%  "
              f"val_loss={res['best_val_loss']:.4f}  "
              f"epoch={res['best_epoch']}  → {ckpt}")
    print(f"\n  Total time: {total_secs:.0f}s ({total_secs/60:.1f} min)")
    print(f"\n  Run the Streamlit app:")
    print(f"    .\\venv\\Scripts\\streamlit.exe run streamlit_app.py")
    print(f"  Then open: http://localhost:8501")
    print(SEP)

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(SEP)
    print("  AI Kidney Stone Detection — Full Pipeline Run")
    print(SEP)
    t_global = time.time()

    step1_generate_dataset()
    step2_split_dataset()
    results, train_secs = step3_train_models()
    print_summary(results, time.time() - t_global)

