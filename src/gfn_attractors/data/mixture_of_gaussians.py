import numpy as np
import torch
from torch import distributions as dist
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from lightning import LightningDataModule
import seaborn as sns
from matplotlib import pyplot as plt


class MixtureOfGaussiansDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        means: np.ndarray,
        covs: np.ndarray,
        mixture_weights: tuple[int, ...] | None = None,
        n_train_samples: int = 2000,
        n_val_samples: int = 200,
        num_workers: int = 0,
        pin_memory: bool = False,
        return_labels: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        self.train_data = MixtureOfGaussiansDataPipe(
            means=self.hparams.means,
            covs=self.hparams.covs,
            mixture_weights=self.hparams.mixture_weights,
            n_samples=self.hparams.n_train_samples,
            return_labels=self.hparams.return_labels
        )
        self.val_data = MixtureOfGaussiansDataPipe(
            means=self.hparams.means,
            covs=self.hparams.covs,
            mixture_weights=self.hparams.mixture_weights,
            n_samples=self.hparams.n_val_samples,
            return_labels=self.hparams.return_labels
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


class GridOfGaussiansDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_dims: int = 2,
        components_per_dim: int = 4,
        scale: float = 1,
        overlap: float = 0.2,
        n_train_samples: int = 2000,
        n_val_samples: int = 200,
        num_workers: int = 0,
        pin_memory: bool = False,
        return_labels: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        self.train_data = GridOfGaussiansDataPipe(
            n_dims=self.hparams.n_dims,
            components_per_dim=self.hparams.components_per_dim,
            overlap=self.hparams.overlap,
            scale=self.hparams.scale,
            n_samples=self.hparams.n_train_samples,
            return_labels=self.hparams.return_labels
        )
        self.val_data = GridOfGaussiansDataPipe(
            n_dims=self.hparams.n_dims,
            components_per_dim=self.hparams.components_per_dim,
            overlap=self.hparams.overlap,
            scale=self.hparams.scale,
            n_samples=self.hparams.n_val_samples,
            return_labels=self.hparams.return_labels
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


class MixtureOfGaussiansDataPipe(MapDataPipe):
    def __init__(
        self,
        means: np.ndarray,
        covs: np.ndarray,
        mixture_weights: tuple[int, ...] | None = None,
        n_samples: int = 2000,
        return_labels: bool = False
    ) -> None:
        super().__init__()

        self.n_components, self.n_dims = means.shape
        assert covs.shape == (self.n_components, self.n_dims, self.n_dims)

        self.means = torch.from_numpy(means).float()
        self.covs = torch.from_numpy(covs).float()
        self.mixture_weights = (
            torch.FloatTensor(mixture_weights)
            if mixture_weights is not None
            else torch.ones(self.n_components)
        )
        self.n_samples = n_samples
        self.return_labels = return_labels

        self.mixture_id = dist.Categorical(self.mixture_weights).sample((self.n_samples, ))
        self.dist = dist.MultivariateNormal(self.means[self.mixture_id], self.covs[self.mixture_id])
        self.data = self.dist.sample()

        self.mixture = dist.MixtureSameFamily(
            dist.Categorical(self.mixture_weights),
            dist.MultivariateNormal(self.means, self.covs),
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index) -> torch.Tensor:
        if self.return_labels:
            return {'data': self.data[index], 'label': self.mixture_id[index]}
        else:
            return self.data[index]

    def visualize(self, ax=None):
        assert self.n_dims == 2
        if ax is None:
            ax = plt.subplot()
        data = self.data.numpy()
        sns.scatterplot(x=data[:, 0], y=data[:, 1], ax=ax)
        return ax


class GridOfGaussiansDataPipe(MixtureOfGaussiansDataPipe):
    def __init__(
        self,
        n_dims: int = 2,
        components_per_dim: int = 4,
        scale: float = 1,
        overlap: float = 0.2,
        n_samples: int = 2000,
        return_labels: bool = False
    ) -> None:
        dim_range = np.linspace(-1, 1, components_per_dim)
        dim_scale = 2 / (components_per_dim - 1)
        cov = (dim_scale * overlap) ** 2

        means = scale * np.array(np.meshgrid(*[dim_range] * n_dims)).T.reshape(-1, n_dims)
        covs = scale * np.array([np.eye(n_dims) * cov] * len(means))
        mixture_weights = np.ones(len(means))

        super().__init__(means, covs, mixture_weights, n_samples, 
                         return_labels=return_labels)
