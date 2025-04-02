import os

import torch
from torch.utils.data import Dataset
import numpy as np

class DrawingDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_paths = []
        self.transform = transform
        self.classes = sorted([file_name for file_name in os.listdir(data_dir) if file_name.endswith(".npy")])

        for class_index, file_name in enumerate(self.classes):
            file_path = os.path.join(data_dir, file_name)
            row_count = np.load(file_path, mmap_mode="r", encoding='latin1', allow_pickle=True).shape[0]

            for row in range(row_count):
                self.data_paths.append((file_path, class_index, row))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path, class_index, row_index = self.data_paths[index]

        data = np.load(data_path, mmap_mode="r", encoding='latin1', allow_pickle=True)[row_index]
        data = (data > 180).astype(np.float32)

        data = torch.tensor(data, dtype=torch.float32).reshape(1, 28, 28)

        if self.transform:
            data = self.transform(data)

        return data, class_index
