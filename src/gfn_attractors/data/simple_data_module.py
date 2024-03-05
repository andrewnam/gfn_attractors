import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchdata.datapipes.map import SequenceWrapper
import pytorch_lightning as pl


class SimpleDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, train_indices=None, num_workers=0, **data):
        super().__init__()
        self.batch_size = batch_size
        if train_indices is None:
            train_indices = torch.arange(len(next(iter(data.values()))))
        self.train_indices = train_indices.sort()[0]
        self.num_workers = num_workers
        self.data = data

        for k, v in data.items():
            setattr(self, k, v)

        self.val_indices = torch.tensor(np.setdiff1d(np.arange(len(self)), train_indices.numpy()), device=train_indices.device)

    def to(self, device):
        self.train_indices = self.train_indices.to(device)
        self.val_indices = self.val_indices.to(device)
        for k, v in self.data.items():
            v = v.to(device)
            self.data[k] = v
            setattr(self, k, v)
        return self

    def train_dataloader(self) -> DataLoader:
        dp = SequenceWrapper(self.train_indices)
        dp = dp.map(lambda i: {k: v[i] for k, v in self.data.items()})
        return DataLoader(dp, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        if len(self.val_indices) == 0:
            return None
        dp = SequenceWrapper(self.val_indices)
        dp = dp.map(lambda i: {k: v[i] for k, v in self.data.items()})
        return DataLoader(dp, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def sample(self, n, train=True, val=True):
        if train and val:
            indices = self.arange(len(self), device=self.device)
        elif train:
            indices = self.train_indices
        elif val:
            indices = self.val_indices
        else:
            raise ValueError('Must sample from at least one of train or val')
        indices = indices[torch.randperm(len(indices))[:n]]
        return self[indices]
    
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.data.items()}

    def __len__(self):
        return len(next(iter(self.data.values())))
