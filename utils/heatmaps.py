# --- utils/heatmaps.py ---
import numpy as np
import torch


def generate_heatmaps(joints, visibility, out_size=56, sigma=2):
    """
    joints: (K, 2) normalized coords [0,1]
    visibility: (K,) 0/1
    out_size: heatmap resolution (H=W)
    sigma: Gaussian spread
    """
    K = joints.shape[0]
    heatmaps = np.zeros((K, out_size, out_size), dtype=np.float32)

    for j in range(K):
        if visibility[j] == 0:
            continue
        x = int(joints[j,0] * out_size)
        y = int(joints[j,1] * out_size)
        if x < 0 or y < 0 or x >= out_size or y >= out_size:
            continue
        xx, yy = np.meshgrid(np.arange(out_size), np.arange(out_size))
        heatmaps[j] = np.exp(-((xx-x)**2 + (yy-y)**2) / (2*sigma**2))
    return heatmaps


def heatmaps_to_coords(heatmaps):
    """
    heatmaps: (B, num_joints, H, W)
    returns: (B, num_joints, 2) in [0,1] normalized coords
    """
    B, J, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B, J, -1)
    idx = heatmaps_reshaped.argmax(dim=2)  # (B, J)

    # Convert flat indices → 2D coords
    y = (idx // W).float()
    x = (idx % W).float()

    coords = torch.stack([x, y], dim=2)  # (B, J, 2)
    coords[..., 0] /= W  # normalize to [0,1]
    coords[..., 1] /= H

    return coords

