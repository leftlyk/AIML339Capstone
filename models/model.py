# --- model.py ---
import torch
import torch.nn as nn
import timm

# LoRA for vision transformers
import loralib as lora

# Utility to patch all attention layers in ViT with LoRA
def apply_lora_to_vit(model, r=4, lora_alpha=1.0):
    """
    Applies LoRA to all attention projection layers and first MLP linear layer in a timm ViT model.
    Args:
        model: timm ViT model
        r: LoRA rank
        lora_alpha: LoRA scaling
    """
    for name, module in model.named_modules():
        # Patch qkv projection in attention
        if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
            module.qkv = lora.Linear(module.qkv.in_features, module.qkv.out_features, r=r, lora_alpha=lora_alpha, bias=module.qkv.bias is not None)
        # Patch output projection in attention
        if hasattr(module, 'proj') and isinstance(module.proj, nn.Linear):
            module.proj = lora.Linear(module.proj.in_features, module.proj.out_features, r=r, lora_alpha=lora_alpha, bias=module.proj.bias is not None)
        # Patch first MLP linear layer (fc1) in transformer block
        if hasattr(module, 'mlp'):
            if hasattr(module.mlp, 'fc1') and isinstance(module.mlp.fc1, nn.Linear):
                module.mlp.fc1 = lora.Linear(
                    module.mlp.fc1.in_features,
                    module.mlp.fc1.out_features,
                    r=r,
                    lora_alpha=lora_alpha,
                    bias=module.mlp.fc1.bias is not None
                )

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


# LoRA-augmented ViT pose model
class ViTPoseHeatmapLoRA(nn.Module):
    def __init__(self, vit_name="vit_small_patch16_224", num_joints=16, hm_size=56, lora_r=4, lora_alpha=1.0):
        super().__init__()
        self.backbone = timm.create_model(vit_name, pretrained=True, features_only=True, out_indices=[-1])
        apply_lora_to_vit(self.backbone, r=lora_r, lora_alpha=lora_alpha)
        c = self.backbone.feature_info[-1]['num_chs']
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_joints, kernel_size=1)
        )

    def forward(self, x):
        feat = self.backbone(x)[-1]
        return self.deconv(feat)


class ViTPoseHeatmapBatchnorm(nn.Module):
    def __init__(self, vit_name="vit_small_patch16_224", num_joints=16, hm_size=56):
        super().__init__()
        self.backbone = timm.create_model(vit_name, pretrained=True, features_only=True, out_indices=[-1])
        c = self.backbone.feature_info[-1]['num_chs']
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c, 256, 4, stride=2, padding=1),  # upsample
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, num_joints, kernel_size=1)
        )

    def forward(self, x):
        feat = self.backbone(x)[-1]   # (B,C,H/16,W/16)
        return self.deconv(feat)      # (B,K,hm_size,hm_size)

class ViTPoseRegression(torch.nn.Module):
    def __init__(self, vit_name="vit_small_patch16_224", num_joints=16):
        super().__init__()
        self.backbone = timm.create_model(vit_name, pretrained=True, features_only=True, out_indices=[-1])
        c = self.backbone.feature_info[-1]['num_chs']
        self.head = torch.nn.Linear(c, num_joints*2)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = feat.mean(dim=[2,3])
        return self.head(feat).view(x.size(0), -1, 2)