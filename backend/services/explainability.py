"""
Explainable AI (Grad-CAM) Service
Generates visual explanations for MobileNetV2, ResNet50, and EfficientNet-B0
using real gradient-weighted class activation mapping (Grad-CAM).
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from config.settings import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    get_device,
)
from backend.services.preprocessing import get_transforms
from backend.services.classification import load_checkpoint, is_model_available


def get_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Find the optimal convolutional target layer for Grad-CAM."""
    arch = architecture.lower()
    if arch == "mobilenet_v2":
        return model.features[-1]
    elif arch == "resnet50":
        return model.layer4[-1]
    elif arch == "efficientnet_b0":
        return model.features[-1]
    else:
        raise ValueError(f"Unsupported architecture for Grad-CAM: {architecture}")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping implementation.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [h1, h2]

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute Grad-CAM heatmap array normalized between 0 and 1.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass with gradient tracking
        input_tensor.requires_grad_(True)
        outputs = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(outputs, dim=1).item())

        score = outputs[0, target_class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations in Grad-CAM hooks.")

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU
        cam = F.relu(cam)

        # Resize to input dimensions
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().cpu().numpy()
        min_v, max_v = cam_np.min(), cam_np.max()
        if max_v > min_v:
            cam_norm = (cam_np - min_v) / (max_v - min_v)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm, target_class_idx


def generate_gradcam_explanation(
    image: Image.Image,
    architecture: str,
    target_class_name: Optional[str] = "Stone",
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Generate complete Grad-CAM explanation including heatmap and blended overlay.
    """
    if not is_model_available(architecture):
        return {
            "success": False,
            "error": f"Model '{architecture}' has not been trained or checkpoint is unavailable.",
        }

    if device is None:
        device = get_device()

    try:
        model, _ = load_checkpoint(architecture, device=device)
        target_layer = get_target_layer(model, architecture)
        grad_cam = GradCAM(model, target_layer)

        transform = get_transforms(split="inference")
        input_tensor = transform(image).unsqueeze(0).to(device)

        target_idx = CLASS_TO_IDX.get(target_class_name, 1) if target_class_name else None
        cam_map, computed_target_idx = grad_cam.generate_cam(input_tensor, target_idx)
        grad_cam.remove_hooks()

        # Build visual representations
        w, h = image.size
        # Resize CAM map to match original image dimensions
        cam_resized = cv2.resize(cam_map, (w, h))
        heatmap_uint8 = np.uint8(255 * cam_resized)

        # Colormap JET
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        heatmap_pil = Image.fromarray(heatmap_rgb)

        # Overlay on original image
        orig_np = np.array(image.convert("RGB"))
        blended = np.uint8(orig_np * (1.0 - alpha) + heatmap_rgb * alpha)
        overlay_pil = Image.fromarray(blended)

        resolved_class_name = IDX_TO_CLASS.get(computed_target_idx, "Unknown")

        return {
            "success": True,
            "architecture": architecture,
            "target_class": resolved_class_name,
            "heatmap_image": heatmap_pil,
            "overlay_image": overlay_pil,
            "disclaimer": (
                "EXPLAINABILITY DISCLAIMER: Grad-CAM visualizes convolutional features that influenced "
                "the neural network's activation for the selected class. It does not constitute radiological "
                "segmentation or prove clinical pathology."
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate Grad-CAM: {str(e)}",
        }
