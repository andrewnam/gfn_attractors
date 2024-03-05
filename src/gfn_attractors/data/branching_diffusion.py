import itertools
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from PIL import Image
from torchdata.datapipes.map import SequenceWrapper
import einops
from ..misc import utils


class BranchingDiffusionDataModule(pl.LightningDataModule):

    def __init__(self, num_features, branching_factor, depth, p_mutate, compression_factor=8, seed=None, batch_size=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = num_features
        self.branching_factor = branching_factor
        self.depth = depth
        self.p_mutate = p_mutate
        self.seed = seed
        self.compression_factor = compression_factor
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.init_data()

    @property
    def num_compressed_tokens(self):
        return 2**self.compression_factor

    @property
    def compressed_length(self):
        return self.num_features // self.compression_factor

    def init_data(self) -> None:
        gen1 = np.zeros((self.branching_factor, self.num_features), dtype=bool)
        indices = utils.get_partition_sizes(self.num_features, self.branching_factor)
        indices.insert(0, 0)
        indices = np.cumsum(indices)
        for a, i, j in zip(gen1, indices[:-1], indices[1:]):
            a[i:j] = True

        generations = [gen1]
        for i in range(1, self.depth):
            gen = self.sample_children(generations[-1], self.branching_factor, self.p_mutate)
            generations.append(gen)
        self.generations = generations
        self.labels = np.array(list(itertools.product(*[range(self.branching_factor)]*self.depth)))

    def sample_children(self, parents, branching_factor, p_mutate):
        """
        parents: np.ndarray of shape (n, num_features)
        """
        parents = np.expand_dims(parents, 1)
        flip = self.rng.binomial(1, p_mutate, size=(len(parents), branching_factor, self.num_features)).astype(bool)
        new_gen = np.logical_xor(parents, flip)
        new_gen = new_gen.reshape(-1, self.num_features)
        return new_gen

    def compress(self, data, compression_factor=None):
        n = len(data)
        if compression_factor is None:
            compression_factor = self.compression_factor
        if isinstance(data, torch.Tensor):
            x = data.view(n, -1, compression_factor)
            bit_values = 2**(torch.arange(x.shape[-1], 0, -1, device=data.device)-1)
            return (x @ bit_values).long()
        else:
            x = data.reshape(n, -1, compression_factor)
            bit_values = 2**(np.arange(x.shape[-1], 0, -1)-1)
            return (x @ bit_values).astype(int)
    
    def create_batch(self, indices, k=1):
        x = self.sample_children(self.generations[-1][indices], k, self.p_mutate)
        labels = self.labels[indices]
        compressed = self.compress(x, self.compression_factor)
        return {
            'index': torch.tensor(indices, dtype=torch.long),
            'x': torch.tensor(x, dtype=torch.float32),
            'compressed': torch.tensor(compressed, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def get_correlation_matrix(self, generation):
        a = self.generations[generation].astype(int)
        m = a @ a.T
        return (m / self.num_features)

    def plot_correlation_matrix(self, generation, size=None):
        im = Image.fromarray(255 * self.get_correlation_matrix(generation))
        if size is not None:
            im = im.resize((size, size))
        return im

    def train_dataloader(self, batch_size=None) -> DataLoader:
        dp = SequenceWrapper(torch.arange(len(self)))
        dp = dp.map(lambda i: self[i])
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(dp, batch_size=batch_size, shuffle=True)
    
    def __getitem__(self, index):
        if isinstance(index, torch.Tensor) and index.numel() == 1:
            return {k: v[0] for k, v in self.__getitems__([index.item()]).items()}
        if not hasattr(index, '__iter__'):
            index = [index]
            return {k: v[0] for k, v in self.__getitems__(index).items()}
        return {k: v for k, v in self.__getitems__(index).items()}

    def __getitems__(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        
        return self.create_batch(indices)
    
    def __len__(self):
        return len(self.generations[-1])



# class BranchingDiffusionDataModule(pl.LightningDataModule):

#     def __init__(self, num_features, branching_factor, depth, p_mutate, compression_factor=8, seed=None, batch_size=128, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_features = num_features
#         self.branching_factor = branching_factor
#         self.depth = depth
#         self.p_mutate = p_mutate
#         self.seed = seed
#         self.compression_factor = compression_factor
#         self.batch_size = batch_size
#         self.rng = np.random.default_rng(seed)
#         self.init_data()

#     @property
#     def num_compressed_tokens(self):
#         return 2**self.compression_factor

#     @property
#     def compressed_length(self):
#         return self.num_features // self.compression_factor

#     def init_data(self) -> None:
#         gen1 = np.zeros((self.branching_factor, self.num_features), dtype=bool)
#         indices = utils.get_partition_sizes(self.num_features, self.branching_factor)
#         indices.insert(0, 0)
#         indices = np.cumsum(indices)
#         for a, i, j in zip(gen1, indices[:-1], indices[1:]):
#             a[i:j] = True

#         generations = [gen1]
#         for i in range(1, self.depth):
#             gen = self.sample_children(generations[-1], self.branching_factor, self.p_mutate)
#             generations.append(gen)
#         self.generations = generations
#         self.labels = np.array(list(itertools.product(*[range(self.branching_factor)]*self.depth)))
#         self.data = self.generations[-1]
#         if self.compression_factor is not None:
#             self.compressed = self.compress(self.generations[-1], self.compression_factor)

#     def sample_children(self, parents, branching_factor, p_mutate):
#         """
#         parents: np.ndarray of shape (n, num_features)
#         """
#         parents = np.expand_dims(parents, 1)
#         flip = self.rng.binomial(1, p_mutate, size=(len(parents), branching_factor, self.num_features)).astype(bool)
#         new_gen = np.logical_xor(parents, flip)
#         new_gen = new_gen.reshape(-1, self.num_features)
#         return new_gen

#     def compress(self, data, compression_factor=None):
#         n = len(data)
#         if compression_factor is None:
#             compression_factor = self.compression_factor
#         if isinstance(data, torch.Tensor):
#             x = data.view(n, -1, compression_factor)
#             bit_values = 2**(torch.arange(x.shape[-1], 0, -1, device=data.device)-1)
#             return (x @ bit_values).long()
#         else:
#             x = data.reshape(n, -1, compression_factor)
#             bit_values = 2**(np.arange(x.shape[-1], 0, -1)-1)
#             return (x @ bit_values).astype(int)
    
#     def create_batch(self, indices):
#         x = self.data[indices]
#         labels = self.labels[indices]
#         compressed = self.compressed[indices]
#         return {
#             'index': torch.tensor(indices, dtype=torch.long),
#             'x': torch.tensor(x, dtype=torch.float32),
#             'compressed': torch.tensor(compressed, dtype=torch.long),
#             'labels': torch.tensor(labels, dtype=torch.long)
#         }
    
#     def get_correlation_matrix(self, generation):
#         a = self.generations[generation].astype(int)
#         m = a @ a.T
#         return (m / self.num_features)

#     def plot_correlation_matrix(self, generation, size=None):
#         im = Image.fromarray(255 * self.get_correlation_matrix(generation))
#         if size is not None:
#             im = im.resize((size, size))
#         return im

#     def train_dataloader(self, batch_size=None) -> DataLoader:
#         dp = SequenceWrapper(torch.arange(len(self)))
#         dp = dp.map(lambda i: self[i])
#         batch_size = self.batch_size if batch_size is None else batch_size
#         return DataLoader(dp, batch_size=batch_size, shuffle=True)
    
#     def __getitem__(self, index):
#         return {k: v[0] for k, v in self.__getitems__([index]).items()}

#     def __getitems__(self, indices):
#         if isinstance(indices, torch.Tensor):
#             indices = indices.tolist()
#         return self.create_batch(indices)
    
#     def __len__(self):
#         return len(self.generations[-1])
