import torch
from torch.utils.data import Dataset
from typing import *


class Subset(Dataset):
    """
    Wrapper for a subset of a dataset.
    """
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


class Part(Subset):
    """
    Wrapper for a part of a dataset.
    """
    def __init__(self, dataset: Dataset, total_parts: int, part: int):
        indices = list(range(len(dataset)))
        indices = indices[part::total_parts]
        super().__init__(dataset, indices)
