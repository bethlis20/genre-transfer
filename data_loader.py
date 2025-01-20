import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

class MusicDataset(Dataset):

    def __init__(self, path):
        super().__init__()
        self.data = np.expand_dims(np.load(path), 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)


def get_dataloader(a, b, training=True, batch_size=16, num_workers=2):
    if training:
        dataset_a = MusicDataset(path=f'data/train/{a}.npy')
        dataset_b = MusicDataset(path=f'data/train/{b}.npy')
    else:
        dataset_a = MusicDataset(path=f'data/test/{a}.npy')
        dataset_b = MusicDataset(path=f'data/test/{b}.npy')

    dataloader_a = DataLoader(dataset_a, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_b = DataLoader(dataset_b, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader_a, dataloader_b
