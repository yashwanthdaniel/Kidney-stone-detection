"""
Unit tests for model construction, checkpoint verification, and inference.
"""
import pytest
from pathlib import Path
import tempfile
import torch
from PIL import Image

from config.settings import CLASS_NAMES, NUM_CLASSES
from backend.services.classification import (
    build_model,
    save_checkpoint,
    load_checkpoint,
    run_single_model_inference,
    SUPPORTED_ARCHITECTURES,
)


def test_model_output_shapes():
    """Verify each model produces [B, 2] output for 2-class classification."""
    dummy_input = torch.randn(2, 3, 224, 224)

    for arch in SUPPORTED_ARCHITECTURES:
        # Build without pretrained weights for fast offline unit testing
        model = build_model(arch, num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 2), f"{arch} output shape mismatch: {output.shape}"


def test_checkpoint_saving_and_loading():
    """Verify checkpoint save and load preserves metadata and state dict."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test_mobilenet.pth"

        model = build_model("mobilenet_v2", num_classes=2, pretrained=False)
        save_checkpoint(
            model=model,
            architecture="mobilenet_v2",
            path=tmp_path,
            epoch=3,
            best_val_acc=94.5,
            best_val_loss=0.15,
            training_config={"lr": 1e-4},
        )

        assert tmp_path.exists()

        # Load back
        loaded_model, cp = load_checkpoint("mobilenet_v2", checkpoint_path=tmp_path, device=torch.device("cpu"))
        assert cp["architecture"] == "mobilenet_v2"
        assert cp["best_val_acc"] == 94.5
        assert cp["class_to_idx"] == {"Non-Stone": 0, "Stone": 1}


def test_architecture_mismatch_rejection():
    """Verify loading ResNet checkpoint into MobileNet architecture raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "resnet_ckpt.pth"

        resnet = build_model("resnet50", num_classes=2, pretrained=False)
        save_checkpoint(
            model=resnet,
            architecture="resnet50",
            path=tmp_path,
            epoch=1,
            best_val_acc=88.0,
            best_val_loss=0.3,
            training_config={},
        )

        # Attempt to load into mobilenet_v2
        with pytest.raises(ValueError, match="Architecture mismatch"):
            load_checkpoint("mobilenet_v2", checkpoint_path=tmp_path, device=torch.device("cpu"))


def test_inference_probability_normalization():
    """Verify inference probabilities sum to 1.0 and canonical classes are returned."""
    model = build_model("mobilenet_v2", num_classes=2, pretrained=False)
    synthetic_img = Image.new("RGB", (224, 224), color=(50, 50, 50))

    res = run_single_model_inference(
        model=model,
        image=synthetic_img,
        architecture="mobilenet_v2",
        device=torch.device("cpu"),
    )

    assert res["available"] is True
    assert res["predicted_class"] in CLASS_NAMES
    assert 0.0 <= res["confidence"] <= 1.0
    p_sum = res["stone_probability"] + res["non_stone_probability"]
    assert abs(p_sum - 1.0) < 1e-4
    assert res["inference_time_ms"] >= 0.0
