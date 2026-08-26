import torch
from torch.utils.data import Dataset
from PIL import Image

class BeeDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx])
        if self.transform:
            img = self.transform(img)
        return img
