import numpy as np

def compute_pckh(preds, targets, visibility, head_indices=(9,8), thresh=0.5):
    """
    preds, targets: [N, K, 2] arrays (in pixels)
    visibility: [N, K] array (1=visible, 0=invisible)
    head_indices: tuple for head length normalization
    thresh: fraction of head segment length
    """
    N, K, _ = preds.shape
    correct, total = 0, 0
    for i in range(N):
        head_len = np.linalg.norm(targets[i, head_indices[0]] - targets[i, head_indices[1]])
        for j in range(K):
            if visibility[i,j] == 0:
                continue
            dist = np.linalg.norm(preds[i,j] - targets[i,j])
            if dist <= thresh * head_len:
                correct += 1
            total += 1
    return correct / total if total>0 else 0.0