# --- utils/heatmaps.py ---
import numpy as np
import cv2

def generate_target_heatmaps(joints, joints_vis, heatmap_size, image_size, sigma=2):
    # joints: (K, 2) in input image coords; returns (K, Hh, Wh)
    K = joints.shape[0]
    target = np.zeros((K, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
    tmp_size = sigma * 3

    feat_stride = np.array(image_size) / np.array(heatmap_size)
    for k in range(K):
        if joints_vis[k] <= 0: 
            continue
        mu_x = int(joints[k][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[k][1] / feat_stride[1] + 0.5)
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
            continue

        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]

        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        target[k, img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            target[k, img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        )
    return target
