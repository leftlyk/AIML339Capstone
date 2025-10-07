# --- model.py ---
import torch
import torch.nn as nn
import timm

class ViTPoseHeatmap(nn.Module):
    def __init__(self, vit_name="vit_small_patch16_224", num_joints=16, hm_size=56):
        super().__init__()
        self.backbone = timm.create_model(vit_name, pretrained=True, features_only=True, out_indices=[-1])
        c = self.backbone.feature_info[-1]['num_chs']
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c, 256, 4, stride=2, padding=1),  # upsample
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_joints, kernel_size=1)
        )

    def forward(self, x):
        feat = self.backbone(x)[-1]   # (B,C,H/16,W/16)
        return self.deconv(feat)      # (B,K,hm_size,hm_size)