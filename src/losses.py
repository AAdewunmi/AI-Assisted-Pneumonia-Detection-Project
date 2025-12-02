"""
Custom loss functions for PneumoDetect.
Includes Focal Loss (for class imbalance mitigation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    Paper: https://arxiv.org/abs/1708.02002
    γ (gamma): focusing parameter, typically 2
    α (alpha): class weighting factor, typically 0.25 for positive class
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return (
            focal_loss.mean()
            if self.reduction == "mean"
            else focal_loss.sum()
        )
