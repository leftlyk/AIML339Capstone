import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
import cv2
from skimage import io, transform
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.heatmaps import generate_heatmaps


class MPIIDataset(Dataset):
    def __init__(self, annotations_path, images_path, img_size=224):
        self.img_size = img_size
        self.images_path = images_path
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.images_path, ann['image'])
        img = cv2.imread(img_path)[..., ::-1]  # BGR → RGB

        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Normalize joints into [0,1] relative to original size
        joints = np.array([(j['x'], j['y']) for j in ann['joints']], dtype=np.float32)
        joints[:,0] = joints[:,0] / orig_w
        joints[:,1] = joints[:,1] / orig_h

        visibility = (joints.sum(axis=1) != 0).astype(np.float32)

        # Generate heatmaps at lower resolution (e.g., 56x56 if img=224)
        heatmaps = generate_heatmaps(joints, visibility, out_size=self.img_size//4)

        # Convert image → torch tensor
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0

        return {
            'image': img,
            'heatmaps': torch.from_numpy(heatmaps),
            'joints': torch.from_numpy(joints),         # still useful for eval/vis
            'visibility': torch.from_numpy(visibility)
        }


