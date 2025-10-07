# --- train.py (outline) ---
import torch, timm, json, os, cv2
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.losses import heatmap_mse_loss
from utils.heatmaps import heatmaps_to_coords
from utils.pckh import compute_pckh
from utils.visualisations import visualize_batch_predictions
from utils.visualisations import visualize_batch_gt
from data.mpii_dataset import MPIIDataset
from models.model import ViTPoseHeatmap


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device='cuda', img_size=224):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        for batch in train_loader:
            imgs = batch['image'].to(device)
            hmaps = batch['heatmaps'].to(device)

            pred = model(imgs)  # (B, J, H, W)
            loss = heatmap_mse_loss(pred, hmaps)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # ---- Validation ----
        model.eval()
        all_preds, all_targets, all_vis = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                hmaps = batch['heatmaps'].to(device)
                joints = batch['joints'].to(device)
                vis = batch['visibility'].to(device)

                pred = model(imgs)

                # loss
                val_loss += heatmap_mse_loss(pred, hmaps).item()

                # decode coords
                coords = heatmaps_to_coords(pred, img_size=img_size)

                all_preds.append(coords * img_size)   # back to pixel space
                all_targets.append(joints * img_size)
                all_vis.append(vis)

        # aggregate
        all_preds_t = torch.cat(all_preds, dim=0)
        all_targets_t = torch.cat(all_targets, dim=0)
        all_vis_t = torch.cat(all_vis, dim=0)

        # compute PCKh
        all_preds_np = all_preds_t.cpu().numpy()
        all_targets_np = all_targets_t.cpu().numpy()
        all_vis_np = all_vis_t.cpu().numpy()
        pckh = compute_pckh(all_preds_np, all_targets_np, all_vis_np)

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, PCKh@0.5: {pckh:.4f}")

        # visualize some predictions
        visualize_batch_predictions(model, val_loader, device=device, img_size=img_size, n_samples=2)

cfg = SimpleNamespace(
    train_annot="../data/annotations/train_annotations_full.json",
    val_annot="../data/annotations/val_annotations_full.json",
    img_root="../data/cropped_persons/",
    batch_size=8,
    vit_name="vit_small_patch16_224",
    lr=1e-3,
    epochs=30
)

train_dataset = MPIIDataset(cfg.train_annot, cfg.img_root)
val_dataset = MPIIDataset(cfg.val_annot, cfg.img_root)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=1)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViTPoseHeatmap(vit_name=cfg.vit_name)

# Use DataParallel if >1 GPU available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model = model.to(device)

visualize_batch_gt(model, val_loader, device='cpu', img_size=224, n_samples=2)

train_model(model, train_loader, val_loader, epochs=cfg.epochs, lr=cfg.lr)