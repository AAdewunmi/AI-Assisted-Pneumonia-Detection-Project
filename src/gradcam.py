"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
for CNN-based explainability in the PneumoDetect project.

This module allows visualization of salient regions influencing
model predictions. It hooks into the last convolutional layer of
a CNN model (e.g., ResNet-50) and computes class activation maps.

Usage example (see notebook 03_gradcam_explainability.ipynb):
--------------------------------------------------------------
from src.gradcam import GradCAM
from torchvision import models, transforms
from PIL import Image
import torch

model = models.resnet50(weights="IMAGENET1K_V1")
model.eval()
gradcam = GradCAM(model, target_layer_name="layer4")

img = Image.open("sample_xray.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
input_tensor = transform(img).unsqueeze(0)

heatmap = gradcam.generate(input_tensor)
gradcam.save_overlay(img, heatmap, output_path="reports/week2_gradcam_samples/sample_overlay.png")
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image


class GradCAM:
    """Computes Grad-CAM visual explanations for CNN classifiers."""

    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        """
        Initialize GradCAM with the target model and layer name.

        Args:
            model (torch.nn.Module): The pretrained CNN model (e.g., ResNet-50).
            target_layer_name (str): The name of the layer to register hooks on (e.g., "layer4").
        """
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Register hooks on the target layer
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                module.register_forward_hook(self._save_activations)
                module.register_backward_hook(self._save_gradients)
                self.target_layer = module
                break
        else:
            raise ValueError(f"Layer {target_layer_name} not found in model.")

    def _save_activations(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input image.

        Args:
            input_tensor (torch.Tensor): The input image tensor (1 x 3 x H x W).
            target_class (int, optional): The class index to visualize. If None, uses the top predicted class.

        Returns:
            np.ndarray: The normalized heatmap (0–255, uint8).
        """
        if input_tensor.ndim != 4:
            raise ValueError("Expected input_tensor of shape (1, 3, H, W)")

        input_tensor.requires_grad = True
        outputs = self.model(input_tensor)
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        loss = outputs[0, target_class]
        self.model.zero_grad()
        loss.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients or activations.")

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)

        # Weight feature maps by importance
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        heatmap = np.uint8(255 * heatmap)
        return heatmap

    @staticmethod
    def overlay_heatmap(
        heatmap: np.ndarray, original_img: Image.Image, alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay the Grad-CAM heatmap onto the original image.

        Args:
            heatmap (np.ndarray): The Grad-CAM heatmap (0–255).
            original_img (PIL.Image.Image): Original RGB image.
            alpha (float): Transparency factor for blending.

        Returns:
            np.ndarray: Combined overlay image (BGR for OpenCV compatibility).
        """
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img = np.array(original_img.resize((heatmap.shape[1], heatmap.shape[0])))
        overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
        return overlay

    def save_overlay(
        self, original_img: Image.Image, heatmap: np.ndarray, output_path: str
    ) -> None:
        """
        Save the overlayed Grad-CAM image to disk.

        Args:
            original_img (PIL.Image.Image): The original input image.
            heatmap (np.ndarray): The Grad-CAM heatmap.
            output_path (str): Path to save the overlay image.
        """
        overlay = self.overlay_heatmap(heatmap, original_img)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)
