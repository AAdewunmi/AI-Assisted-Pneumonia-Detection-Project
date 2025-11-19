"""
src/model.py
------------
Defines ResNet-50 architectures for the PneumoDetect project:
- Baseline model (frozen backbone)
- Fine-tuned model (last two residual blocks unfrozen)
All models share a consistent FC head layout for checkpoint compatibility.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_resnet50_baseline(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Build a ResNet-50 model pretrained on ImageNet for binary classification.

    Args:
        num_classes (int): Number of output classes. Default = 2.
        freeze_backbone (bool): If True, freezes all convolutional layers.

    Returns:
        nn.Module: Configured ResNet-50 model ready for inference or training.
    """
    # Load pretrained ResNet-50
    model = models.resnet50(weights="IMAGENET1K_V1")

    # Optionally freeze convolutional layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace fully connected layer (consistent with training checkpoints)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_resnet50_finetuned(num_classes: int = 2) -> nn.Module:
    """
    Build a fine-tuned ResNet-50:
    - Unfreezes the last two residual blocks (layer3, layer4)
    - Keeps earlier layers frozen
    - Uses same FC head format for checkpoint compatibility
    """
    model = models.resnet50(weights="IMAGENET1K_V1")

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last two residual blocks
    for layer_name in ["layer3", "layer4"]:
        for param in getattr(model, layer_name).parameters():
            param.requires_grad = True

    # Consistent FC head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    # Sanity check for both models
    x = torch.randn(4, 3, 224, 224)
    base = build_resnet50_baseline()
    fine = build_resnet50_finetuned()
    print("Baseline output:", base(x).shape)
    print("Fine-tuned output:", fine(x).shape)
