import numpy as np
import torch
from torch import distributions as dist
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from torchdata.datapipes.map import SequenceWrapper
import pandas as pd


class GridOfGaussiansDataModule(LightningDataModule):

    def __init__(self, 
                 batch_size,
                 n_dims=2,
                 components_per_dim=4,
                 scale=1, 
                 overlap=0.2,
                 dim_projection=None,
                 n_train_samples=2000, 
                 n_val_samples=200):
        super().__init__()
        self.save_hyperparameters(logger=False)

        dim_range = np.linspace(-1, 1, components_per_dim)
        dim_scale = 2 / (components_per_dim - 1)
        cov = (dim_scale * overlap) ** 2

        means = scale * np.array(np.meshgrid(*[dim_range] * n_dims)).T.reshape(-1, n_dims)
        covs = scale * np.array([np.eye(n_dims) * cov] * len(means))
        self.means = torch.tensor(means, dtype=torch.float)
        self.covs = torch.tensor(covs, dtype=torch.float)
        weights = torch.ones(len(self.means))
        self.mixture_id = dist.Categorical(weights).sample((n_train_samples + n_val_samples, ))
        self.dist = dist.MultivariateNormal(self.means[self.mixture_id], self.covs[self.mixture_id])
        self.gaussian_data = self.dist.sample()

        if dim_projection is not None:
            self.projection = torch.randn(n_dims, dim_projection)
            self.data = self.gaussian_data @ self.projection
        else:
            self.data = self.gaussian_data

        self.df_data = pd.DataFrame(zip(range(len(self.data)), self.mixture_id.numpy()),
                                    columns=['idx', 'mixture_id'])

    def get_dataloader_item(self, index):
        item = {'index': index,
                'data': self.data[index],
                'gaussian_data': self.gaussian_data[index],
                'label': self.mixture_id[index]}
        return item
    
    def train_dataloader(self, batch_size=None) -> DataLoader:
        dp = SequenceWrapper(self.hparams.n_train_samples)
        dp = dp.map(lambda i: self.get_dataloader_item(i))
        batch_size = self.hparams.batch_size if batch_size is None else batch_size
        return DataLoader(dp, batch_size=batch_size, shuffle=True)
    
    def val_dataloader(self, batch_size=None) -> DataLoader:
        dp = SequenceWrapper(self.hparams.n_val_samples)
        dp = dp.map(lambda i: self.get_dataloader_item(self.hparams.n_train_samples + i))
        batch_size = self.hparams.batch_size if batch_size is None else batch_size
        return DataLoader(dp, batch_size=batch_size, shuffle=True)
