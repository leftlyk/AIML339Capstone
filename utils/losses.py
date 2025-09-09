# --- losses.py ---
import torch
import torch.nn as nn

class HeatmapMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super().__init__()
        self.crit = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, pred, target, target_weight=None):
        # pred, target: (B, K, Hh, Wh), target_weight: (B, K, 1)
        if self.use_target_weight and target_weight is not None:
            loss = ((pred - target) ** 2) * target_weight.unsqueeze(-1).unsqueeze(-1)
            return loss.mean()
        return self.crit(pred, target)
