import itertools
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from PIL import Image
from functools import cached_property
from torchdata.datapipes.map import SequenceWrapper
import einops
import networkx as nx
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
    
    @property
    def data(self):
        return self.generations[-1]

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
        x = self.sample_children(self.data[indices], k, self.p_mutate)
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
        batch_size = self.batch_size if batch_size is None else batch_size
        indices = torch.arange(len(self))
        if batch_size > len(indices):
            indices = indices.repeat(batch_size // len(indices))
        dp = SequenceWrapper(indices)
        dp = dp.map(lambda i: self[i])
        return DataLoader(dp, batch_size=batch_size, shuffle=True)
    
    def render(self, x, scale=1):
        """
        x: np.ndarray of shape (n, num_features)
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.repeat(x, scale, axis=0)
        x = np.repeat(x, scale, axis=1)
        return Image.fromarray((x * 255))
    
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
        return len(self.data)
    


class BinarySplitDataModule(BranchingDiffusionDataModule):

    def __init__(self, depth, repeat=1, sample_ancestors=False, min_test_depth=3, compression_factor=8, seed=None, batch_size=128, *args, **kwargs):
        assert min_test_depth > 0
        self.repeat = repeat
        self.sample_ancestors = sample_ancestors
        self.min_test_depth = min_test_depth
        super().__init__(num_features=repeat * 2**depth, branching_factor=2, depth=depth, p_mutate=0, 
                         compression_factor=compression_factor, seed=seed, batch_size=batch_size, *args, **kwargs)
        
    @cached_property
    def graph(self):
        graph = nx.DiGraph()
        graph.add_node('', gen=0, index=None, test=False)
        for i, label in enumerate(self.labels):
            gen = (label > 0).sum()
            k = ''.join(label[:gen].astype(str))
            graph.add_node(k, gen=gen, index=i, test=i in self.test_indices)
            graph.add_edge(k[:-1], k)
        return graph
        
    def init_data(self) -> None:
        n = 2 ** (self.depth)
        generations = [np.ones((1, n), dtype=np.float32)]
        prototypes = [np.ones((1, n), dtype=np.float32)]
        for k in 2 ** np.arange(1, self.depth+1):
            a = np.eye(k)
            a = np.repeat(a, n // k, axis=1)
            prototypes.append(np.repeat(a, self.repeat, axis=1))
            a = a + np.repeat(generations[-1], 2, axis=0)
            generations.append(a)
        generations = [np.repeat(a, self.repeat, axis=1) for a in generations]
        self.prototypes = prototypes

        labels = []
        for i in range(1 + self.depth):
            l = itertools.product(*[range(1, 3)]*i)
            l = np.array([list(x) + [0] * (self.depth - i) for x in l])
            labels.append(l)
        self.generations = generations

        if self.sample_ancestors:
            sample_probs = np.concatenate(self.generations[1:])
            self.labels = np.concatenate(labels[1:])
        else:
            sample_probs = self.generations[-1]
            self.labels = labels[-1]
        sample_probs = (2**(sample_probs))
        self.sample_probs = sample_probs / (1+sample_probs.max(-1)).reshape(-1, 1)
        self.split_data()

    def split_data(self):
        test_indices = []
        for i in range(self.min_test_depth-1, self.depth):
            if i < self.depth - 1:
                indices = np.arange(len(self))[(self.labels[:, i] > 0) & (self.labels[:, i+1] == 0)]
            else:
                indices = np.arange(len(self))[(self.labels[:, i] > 0)]

            # Remove samples that whose ancestors are already in the test set
            if i > 0:
                prev = self.labels[test_indices][:,:i].reshape(-1, 1, i)
                cur = self.labels[indices][:,:i].reshape(1, -1, i,)
                match = (prev == cur) | (prev == 0)
                match = match.all(-1).any(0)
                test_indices.extend(indices[match])
                test_index = self.rng.choice(indices[~match])
            else:
                test_index = self.rng.choice(indices)
            test_indices.append(test_index)
        self.test_indices = np.array(test_indices)
        self.train_indices = np.setdiff1d(np.arange(len(self)), test_indices)

    def create_batch(self, indices=None, prototype=False, n=None, device='cpu'):
        if indices is None:
            indices = np.arange(len(self))
        if n is not None:
            indices = np.repeat(indices, n)
        if prototype:
            if self.sample_ancestors:
                prototypes = np.concatenate(self.prototypes[1:])
            else:
                prototypes = self.prototypes[-1]
            x = prototypes[indices]
        else:
            p = self.sample_probs[indices]
            x = (self.rng.random(p.shape) < p).astype(float)
        labels = self.labels[indices]
        compressed = self.compress(x, self.compression_factor)
        return {
            'index': torch.tensor(indices, dtype=torch.long, device=device),
            'x': torch.tensor(x, dtype=torch.float32, device=device),
            'compressed': torch.tensor(compressed, dtype=torch.long, device=device),
            'labels': torch.tensor(labels, dtype=torch.long, device=device)
        }

    def train_dataloader(self, batch_size=None) -> DataLoader:
        batch_size = self.batch_size if batch_size is None else batch_size
        indices = self.train_indices
        if batch_size > len(indices):
            indices = indices.repeat(batch_size // len(indices))
        dp = SequenceWrapper(indices)
        dp = dp.map(lambda i: self[i])
        return DataLoader(dp, batch_size=batch_size, shuffle=True)

    @cached_property
    def data(self):
        return self.sample_probs
