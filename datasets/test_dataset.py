from torch.utils.data import Dataset
import torch
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class TestDataset(Dataset):
    def __init__(self, mean, std, size, dim):
        self.mean = mean
        self.std = std
        self.size = size
        self.dim = dim

    def __getitem__(self, index):
        return torch.randn(self.dim) * self.std + self.mean, torch.randn(self.dim) * self.std + self.mean

    def __len__(self) -> int:
        return self.size
