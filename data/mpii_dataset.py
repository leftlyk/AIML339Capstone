import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from skimage import io, transform

class MPIIDataset(Dataset):
    def __init__(self, annotations_path, images_path, transform=None):
        self.images_path = images_path
        try:
            with open(annotations_path, 'r') as f:
                self.annotations = json.load(f)
        except Exception as e:
            print(f"File Error: {e}")
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_path,
                                self.annotations[idx]['image'])
        image = io.imread(img_name)

        joints = [(joint['x'], joint['y']) for joint in self.annotations[idx]['joints']]
        joints = np.array([joints], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'joints': joints}

        if self.transform:
            sample = self.transform(sample)

        return sample

