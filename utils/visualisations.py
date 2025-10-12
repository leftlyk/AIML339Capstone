import matplotlib.pyplot as plt
import numpy as np
import torch

from .heatmaps import heatmaps_to_coords

def visualize_joints_on_image(img, joints, joint_names):
    """
    img         : (H, W, 3) np.uint8 RGB image
    joints      : (K, 2) np.array normalized [0,1] or absolute coords
    joint_names : list of joint names, length K
    """
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img)

    K = len(joint_names)
    colors = plt.cm.tab20(np.linspace(0,1,K))  # 20 distinct colors

    for j, (x, y) in enumerate(joints):
        if x == 0 and y == 0:   # skip invisible
            continue
        ax.scatter(x, y, s=30, color=colors[j], marker='o')
        ax.text(x+2, y-2, joint_names[j], color=colors[j], fontsize=8)

    ax.axis("off")
    plt.show()



def visualize_joints_with_preds(img, joints_gt, joints_pred, joint_names):
    """
    img         : torch.Tensor (3,H,W) [0,1] or np.ndarray (H,W,3) uint8
    joints_gt   : (K, 2) np.array ground truth coords (in px)
    joints_pred : (K, 2) np.array predicted coords (in px)
    joint_names : list of joint names, length K
    """

    # Convert tensor → numpy image if needed
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dim() == 3 and img.shape[0] == 3:   # (3,H,W)
            img = img.permute(1, 2, 0)
        img = img.numpy()
        # Scale if normalized [0,1]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img)

    K = len(joint_names)
    colors = plt.cm.tab20(np.linspace(0,1,K))  # distinct colors

    for j in range(K):
        x_gt, y_gt = joints_gt[j]
        x_pr, y_pr = joints_pred[j]

        # Skip invisible
        if (x_gt == 0 and y_gt == 0):
            continue

        # GT (circle + label)
        ax.scatter(x_gt, y_gt, s=40, color=colors[j], marker='o',
                   edgecolors='k', linewidths=0.5, zorder=3)
        ax.text(x_gt+2, y_gt-2, joint_names[j], color=colors[j],
                fontsize=7, zorder=4)

        # Prediction (X)
        ax.scatter(x_pr, y_pr, s=40, color=colors[j], marker='x', zorder=3)

        # Error line
        ax.plot([x_gt, x_pr], [y_gt, y_pr], color=colors[j],
                linestyle="--", linewidth=1, zorder=2)

    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_img_with_joints(img, pred_joints, gt_joints=None, title=None):
    """
    img         : (H, W, 3) numpy float [0,1]
    pred_joints : (J, 2) predicted pixel coords (x,y)
    gt_joints   : (J, 2) ground-truth pixel coords (x,y), optional
    """
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img)

    # predictions
    ax.scatter(pred_joints[:,0], pred_joints[:,1], c='r', s=20, label='Pred', alpha=0.7)

    # ground truth
    if gt_joints is not None:
        ax.scatter(gt_joints[:,0], gt_joints[:,1], c='g', s=20, label='GT', alpha=0.7)

    if title:
        ax.set_title(title)
    ax.axis('off')
    ax.legend()
    plt.show()


def visualize_batch_predictions(model, loader, device='cuda', img_size=224, n_samples=2):
    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'][:n_samples].to(device)
            gt_joints = batch['joints'][:n_samples].cpu().numpy() * img_size

            # predict
            pred_hmaps = model(imgs)
            pred_coords = heatmaps_to_coords(pred_hmaps).cpu().numpy() * img_size

            imgs = imgs.cpu().permute(0,2,3,1).numpy()

            for i in range(n_samples):
                plot_img_with_joints(
                    imgs[i],
                    pred_coords[i],
                    gt_joints[i],
                    title=f"Sample {i+1}"
                )
            break

def visualize_batch_gt(model, val_loader, device='cuda', img_size=224, n_samples=2):
    """
    Show n_samples from a batch in the val_loader.
    """
    mpii_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip',
                   6: 'pelvis', 7: 'thorax', 8: 'upper_neck', 11: 'relb', 10: 'rwri', 9: 'head',
                   12: 'rsho', 13: 'lsho', 14: 'lelb', 15: 'lwri'}

    model.eval()
    import random
    with torch.no_grad():
        # take one random batch
        batch = next(iter(val_loader))
        imgs = batch['image'].to(device)
        joints_gt = (batch['joints'].numpy())
        visibility = batch['visibility'].numpy()

        indices = random.sample(range(imgs.size(0)), min(n_samples, imgs.size(0)))
        for i in indices:
            img = imgs[i].cpu().permute(1,2,0).numpy()
            print(img.shape)
            print(joints_gt[i])
            joints_px = joints_gt[i] * img_size  # shape (16,2)
            visualize_joints_on_image(img, joints_px, mpii_idx_to_jnt)