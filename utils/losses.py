# --- losses.py ---
import torch
import torch.nn as nn


def heatmap_mse_loss(pred, target):
    return ((pred - target) ** 2).mean()