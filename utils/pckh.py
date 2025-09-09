import numpy as np

def compute_pckh(preds, targets, head_indices=(9, 8), thresh=0.5, visibility=None):
    """
    preds, targets: [N, num_joints, 2] arrays (x, y)
    head_indices: tuple of (head, upper_neck) joint indices
    thresh: fraction of head segment length
    visibility: [N, num_joints] array or None
    """
    N, num_joints, _ = preds.shape
    correct = 0
    total = 0
    for i in range(N):
        head = targets[i, head_indices[0]]
        neck = targets[i, head_indices[1]]
        head_len = np.linalg.norm(head - neck)
        for j in range(num_joints):
            if visibility is not None and visibility[i, j] == 0:
                continue
            dist = np.linalg.norm(preds[i, j] - targets[i, j])
            if head_len > 0 and dist < thresh * head_len:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0