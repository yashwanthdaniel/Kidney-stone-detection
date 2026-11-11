"""
Sequential Training Runner — MobileNetV2, ResNet50, EfficientNet-B0
Runs in the terminal with live epoch-by-epoch progress output.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.train_classifier import train_single_classifier
from backend.services.classification import SUPPORTED_ARCHITECTURES

EPOCHS        = 15
BATCH_SIZE    = 16
LEARNING_RATE = 1e-4
PATIENCE      = 5


def progress_callback(info: dict):
    improved = "✅" if info["improved"] else "  "
    print(
        f"  {improved} Epoch {info['epoch']:>2}/{info['total_epochs']} | "
        f"Train loss: {info['train_loss']:.4f}  acc: {info['train_acc']:5.1f}% | "
        f"Val loss: {info['val_loss']:.4f}  acc: {info['val_acc']:5.1f}% | "
        f"Best val acc: {info['best_val_acc']:5.1f}%  patience: {info['patience_counter']}/{PATIENCE}"
    )
    sys.stdout.flush()


def main():
    print("=" * 70)
    print("  AI Kidney Stone Detection — Model Training")
    print("=" * 70)
    print(f"  Architectures : {', '.join(SUPPORTED_ARCHITECTURES)}")
    print(f"  Epochs        : {EPOCHS}  |  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate : {LEARNING_RATE}  |  Early-stop patience: {PATIENCE}")
    print("=" * 70)

    overall_start = time.time()

    for arch in SUPPORTED_ARCHITECTURES:
        print(f"\n{'─'*70}")
        print(f"  🧠 Training: {arch.upper().replace('_',' ')}")
        print(f"{'─'*70}")

        t0 = time.time()
        result = train_single_classifier(
            architecture=arch,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            patience=PATIENCE,
            progress_callback=progress_callback,
        )
        elapsed = time.time() - t0

        print(f"\n  ✔ Done in {elapsed:.0f}s")
        print(f"    Best epoch      : {result['best_epoch']}")
        print(f"    Best val acc    : {result['best_val_acc']}%")
        print(f"    Best val loss   : {result['best_val_loss']}")
        print(f"    Checkpoint      : {result['checkpoint_path']}")

    total = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"  🎉 All models trained in {total:.0f}s ({total/60:.1f} min)")
    print(f"  Open the app and go to Model Comparison / CT Image Analysis")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

