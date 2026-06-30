import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class MeasurementDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data['X'].astype(np.float32))
        self.y = torch.from_numpy(data['y'].astype(np.float32))
        print(f"Loaded {len(self.X):,} samples from {npz_path}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(npz_path, val_frac=0.15, batch_size=256, seed=42):
    dataset = MeasurementDataset(npz_path)
    n_val   = int(len(dataset) * val_frac)
    n_train = len(dataset) - n_val
    gen     = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False))
