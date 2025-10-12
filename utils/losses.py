# --- losses.py ---
import torch
import torch.nn as nn


def heatmap_mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

def heatmap_mse_loss_per_joint(pred, target):
    # MSE per joint, sum joints, average over batch
    loss = ((pred - target) ** 2).mean(dim=(2, 3)).sum(dim=1).mean()
    return loss

def masked_mse(pred, target, mask):
    loss = ((pred - target)**2).sum(dim=2)
    loss = loss * mask
    return loss.sum() / mask.sum()