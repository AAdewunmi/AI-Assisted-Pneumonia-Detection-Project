"""
src/gradcam.py
--------------
Implements Grad-CAM explainability for CNN models (e.g., ResNet).
Generates and overlays visual saliency maps showing which regions most influenced
a model’s prediction. Supports standalone execution and integration with notebooks.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Optional


class GradCAM:
    """
    Grad-CAM implementation for visualizing important regions in CNN decisions.

    Attributes:
        model (torch.nn.Module): The model being analyzed.
        target_layer_name (str): The name of the convolutional layer to hook.
        gradients (torch.Tensor | None): Captured gradients.
        activations (torch.Tensor | None): Captured forward activations.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        """
        Initialize the GradCAM module and register forward/backward hooks.

        Args:
            model (torch.nn.Module): The CNN model.
            target_layer_name (str): The name of the convolutional layer to hook.
        """
        self.model = model.eval()
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Attach forward and backward hooks to the target layer."""
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(self._forward_hook)
                module.register_full_backward_hook(self._backward_hook)
                return
        raise ValueError(f"Layer {self.target_layer_name} not found in model")

    def _forward_hook(self, module, inputs, output):
        """Capture forward activations."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture backward gradients."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        """
        Generate a Grad-CAM heatmap tensor (0–1 normalized).

        Args:
            input_tensor (torch.Tensor): Input image (1, 3, H, W).
            target_class (int, optional): Class index to visualize.

        Returns:
            torch.Tensor: 2D heatmap (values between 0–1).
        """
        if input_tensor.ndim != 4:
            raise ValueError("Expected input_tensor of shape (1, 3, H, W)")

        # Ensure gradients are enabled and detached safely
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        outputs = self.model(input_tensor)
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        loss = outputs[0, target_class]
        self.model.zero_grad()
        loss.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients or activations")

        # Compute Grad-CAM weights
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]

        # Average across channels → ReLU → Normalize
        heatmap = torch.mean(activations, dim=0)
        heatmap = F.relu(heatmap)
        heatmap -= heatmap.min()
        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        return heatmap.detach().cpu()

    @staticmethod
    def overlay_heatmap(img: np.ndarray, heatmap: torch.Tensor, alpha: float = 0.5) -> np.ndarray:
        """
        Overlay a Grad-CAM heatmap onto an image.

        Args:
            img (np.ndarray): Base image (H, W, 3).
            heatmap (torch.Tensor): 2D normalized tensor (0–1).
            alpha (float): Blend ratio.

        Returns:
            np.ndarray: Combined overlay image (uint8).
        """
        if not isinstance(heatmap, torch.Tensor):
            raise TypeError("heatmap must be a torch.Tensor")

        h, w = img.shape[:2]
        heatmap_np = (heatmap.numpy() * 255).astype(np.uint8)

        # Resize heatmap to match input image
        heatmap_np = cv2.resize(heatmap_np, (w, h))

        heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Blend them safely
        overlay = cv2.addWeighted(img, alpha, heatmap_color, 1 - alpha, 0)
        return overlay


# -------------------------------------------------------------------------------------
# Standalone execution demo
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    from torchvision import models

    model = models.resnet50(weights=None)
    cam = GradCAM(model, target_layer_name="layer4")

    dummy = torch.randn(1, 3, 224, 224, requires_grad=True)
    heatmap = cam.generate(dummy)

    # Convert dummy tensor → RGB image
    img = (dummy.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    overlay = GradCAM.overlay_heatmap(img, heatmap)

    output_dir = Path("reports/week2_gradcam_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "synthetic_demo_overlay.png"
    cv2.imwrite(str(out_path), overlay)
    print(f"Saved synthetic Grad-CAM overlay → {out_path.resolve()}")
