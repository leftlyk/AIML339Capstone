# --- model.py ---
import torch
import torch.nn as nn
import timm

class ViTPoseHead(nn.Module):
    def __init__(self, in_ch, num_joints=16, up_channels=(512,256,256), heatmap_size=(64,64)):
        super().__init__()
        layers = []
        curr_ch = in_ch
        for ch in up_channels:
            layers += [
                nn.ConvTranspose2d(curr_ch, ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
            curr_ch = ch
        self.deconv = nn.Sequential(*layers)
        self.final = nn.Conv2d(curr_ch, num_joints, kernel_size=1)
        self.heatmap_size = heatmap_size

    def forward(self, feat):  # feat: (B, C, H', W')
        x = self.deconv(feat)
        return self.final(x)  # (B, K, Hh, Wh)

class ViTPose(nn.Module):
    def __init__(self, vit_name="vit_base_patch16_224", num_joints=16):
        super().__init__()
        self.backbone = timm.create_model(vit_name, pretrained=True, features_only=True, out_indices=[-1])
        c_out = self.backbone.feature_info[-1]['num_chs']  # e.g., 768
        self.head = ViTPoseHead(in_ch=c_out, num_joints=num_joints)

    def forward(self, x):
        feats = self.backbone(x)[-1]  # (B, C, H/16, W/16) for patch16
        heatmaps = self.head(feats)
        return heatmaps
