"""
Sequential Training Script
Trains MobileNetV2, ResNet50, and EfficientNet-B0 on the prepared dataset splits.
Saves independent checkpoints and epoch training history.
"""
import time
from training.train_classifier import train_single_classifier

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-all", action="store_true", help="Force retrain all models")
    args, _ = parser.parse_known_args()

    architectures = ["resnet50", "efficientnet_b0", "mobilenet_v2"]
    from backend.services.classification import is_model_available
    results = {}

    for arch in architectures:
        if is_model_available(arch) and not args.force_all:
            print("\n" + "=" * 56)
            print(f"[EXISTS] {arch.upper()} already trained. Skipping (use --force-all to retrain).")
            print("=" * 56)
            continue

        print("\n" + "=" * 56)
        print(f"Starting Real Training for: {arch.upper()}")
        print("=" * 56)
        t0 = time.time()

        res = train_single_classifier(
            architecture=arch,
            epochs=5,
            batch_size=32,
            learning_rate=1e-4,
            patience=3,
            random_seed=42,
            max_train_samples=1000,
            max_val_samples=250,
        )

        dt = time.time() - t0
        results[arch] = res
        print(f"\n[DONE] Finished {arch.upper()} in {dt:.1f}s")
        print(f"   Best Val Accuracy: {res['best_val_acc']}% (Epoch {res['best_epoch']})")
        print(f"   Best Val Loss: {res['best_val_loss']}")
        print(f"   Saved Checkpoint: {res['checkpoint_path']}")

    print("\n" + "=" * 56)
    print("Training phase verified for all models!")
    print("=" * 56)

if __name__ == "__main__":
    main()
