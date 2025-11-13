"""
src/model.py
------------
Defines the baseline transfer-learning model using a pretrained ResNet-50.
Freezes the feature extractor and replaces the classification head with
a new linear layer for binary pneumonia classification (Normal vs Pneumonia).
"""

import torch
import torch.nn as nn
from torchvision import models


def build_resnet50_baseline(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Builds a ResNet-50 model pretrained on ImageNet for binary classification.

    Args:
        num_classes (int): Number of output classes. Default = 2.
        freeze_backbone (bool): If True, all convolutional layers are frozen.

    Returns:
        nn.Module: Configured ResNet-50 model.
    """
    # Load pretrained ResNet-50 backbone
    model = models.resnet50(weights="IMAGENET1K_V1")

    # Optionally freeze convolutional backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace fully connected (classification) head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes),
        nn.LogSoftmax(dim=1)
    )

    return model


if __name__ == "__main__":
    # Quick sanity check
    model = build_resnet50_baseline()
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
