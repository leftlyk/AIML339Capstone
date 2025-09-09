# --- train.py (outline) ---
import sys
import torch
import os
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from torch.utils.data import DataLoader
from models.model import ViTPose
from utils.losses import HeatmapMSELoss
from utils.pckh import compute_pckh
from data.mpii_dataset import MPIIDataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import numpy as np

def validate(model, val_loader, criterion, device='cuda'):
    model.eval()
    val_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['image'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            tw = batch['target_weight'].to(device, non_blocking=True)
            pred = model(imgs)
            loss = criterion(pred, target, tw)
            val_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    avg_loss = val_loss / n_samples
    all_preds = np.concatenate(all_preds, axis=0)  # [N, num_joints, 2]
    all_targets = np.concatenate(all_targets, axis=0)  # [N, num_joints, 2]
    # If you have visibility masks, also concatenate them
    # all_vis = np.concatenate(all_vis, axis=0)  # [N, num_joints]

    pckh = compute_pckh(all_preds, all_targets, head_indices=(9, 8), thresh=0.5)
    print(f"PCKh@0.5: {pckh:.4f}")
    return avg_loss


def train(cfg):
    train_set = MPIIDataset(cfg.train_annot, cfg.img_root)
    val_set   = MPIIDataset(cfg.val_annot, cfg.img_root)
    train_loader = DataLoader(train_set, batch_size=cfg.bs, shuffle=True, num_workers=cfg.nw, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=cfg.bs, shuffle=False, num_workers=cfg.nw, pin_memory=True)

    model = ViTPose(vit_name=cfg.vit_name, num_joints=16).cuda()
    criterion = HeatmapMSELoss(use_target_weight=True)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = GradScaler()

    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            imgs = batch['image'].cuda(non_blocking=True)
            target = batch['target'].cuda(non_blocking=True)
            tw = batch['target_weight'].cuda(non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast():
                pred = model(imgs)
                loss = criterion(pred, target, tw)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{cfg.epochs} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{cfg.model_path}/best_model.pth")
            print("Best model saved.")


cfg = SimpleNamespace(**{
    "train_annot": "data/annotations/train_annotations.json",
    "val_annot": "data/annotations/val_annotations.json",
    "img_root": "data/cropped_persons/",
    "model_path": "model_weights",
    "nw": 1,
    "bs": 16,
    "vit_name": "vit_base_patch16_224",
    "lr": 0.001,
    "wd": 0.01,
    "epochs": 50,
})

train(cfg)