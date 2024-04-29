from abc import ABC, abstractmethod
import numpy as np
import torch
import pandas as pd
from plotnine import *


class EvaluationAttractorsModel(ABC):

    def __init__(self, model, data_module, seed=0):
        self.model = model
        self.data_module = data_module
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    def config(self):
        return self.model.config
    
    @property
    def device(self):
        return self.model.device
    
    def sample_training_batch(self, n):
        """
        Should at least contain 'index' and 'x' tensors.
        """
        indices = self.rng.choice(self.data_module.train_indices, n, replace=False)
        batch = self.data_module.create_batch(indices)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch

    def sample_validation_batch(self, n):
        """
        Should at least contain 'index' and 'x' tensors.
        """
        indices = self.rng.choice(self.data_module.valid_indices, n, replace=False)
        batch = self.data_module.create_batch(indices)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch

    @abstractmethod
    def get_z0(self, x):
        """
        x: tensor of shape [batch_size, ...]
        returns tensor of shape [batch_size, dim_z]
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample_forward_trajectory(self, x, z0, num_steps=None, deterministic=False):
        """
        x: tensor of shape [batch_size, ...]
        z0: tensor of shape [batch_size, dim_z]
        returns tensor of shape [batch_size, num_steps, dim_z]
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample_backward_trajectory(self, z, z0, num_steps=None, deterministic=False):
        """
        z: tensor of shape [batch_size, dim_z]
        z0: tensor of shape [batch_size, dim_z]
        returns tensor of shape [batch_size, num_steps, dim_z]
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample_w(self, z, z0=None, temperature=1., argmax=False):
        """
        z: tensor of shape [batch_size, dim_z]
        z0: tensor of shape [batch_size, dim_z]
        target: tensor of shape [batch_size]
        returns 
            w: tensor of shape [batch_size, max_w_length]
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_z_hat(self, w):
        """
        w: tensor of shape [batch_size, max_w_length]
        returns tensor of shape [batch_size, dim_z]
        """
        raise NotImplementedError
    
    def to(self, device):
        self.model = self.model.to(device)

    def plot_distances(self, n=500, num_steps=100, deterministic=False):
        batch_train = self.sample_training_batch(n)
        batch_valid = self.sample_validation_batch(n)
        x = torch.cat([batch_train['x'], batch_valid['x']], dim=0)
        with torch.no_grad():
            z0 = self.get_z0(x)
            z_traj = self.sample_forward_trajectory(z0, num_steps, deterministic=deterministic)
            w = self.sample_w(z_traj[:,self.config.num_steps], z0)
            z_hat = self.get_z_hat(w)
        distances = (z_traj - z_hat.unsqueeze(1)).norm(dim=-1).view(n, 2, -1).mean(0).cpu().numpy()
        df = pd.DataFrame(distances.T, columns=['train', 'valid'])
        df['t'] = range(len(df))
        df = df.melt(id_vars='t', value_vars=['train', 'valid'], var_name='split', value_name='distance')

        p = (ggplot(df, aes(x='t', y='distance', color='split')) 
        + geom_hline(yintercept=((self.config.dim_z * self.config.attractor_sd**2)**.5), linetype='dashed', color='black')
        + geom_vline(xintercept=self.config.num_steps, linetype='dashed', color='black')
        + geom_line(size=1)
        + coord_cartesian(ylim=(0, distances.max()))
        + labs(x='Step', y='Distance')
        + theme_bw()
        )
        return p, df

    def plot_speed(self, n=500, num_steps=100, deterministic=False):
        batch_train = self.sample_training_batch(n)
        batch_valid = self.sample_validation_batch(n)
        x = torch.cat([batch_train['x'], batch_valid['x']], dim=0)
        with torch.no_grad():
            z0 = self.get_z0(x)
            z_traj = self.sample_forward_trajectory(z0, num_steps, deterministic=deterministic)
        speed = (z_traj[:,1:] - z_traj[:,:-1]).norm(dim=-1).view(n, 2, -1).mean(0).cpu().numpy()
        df = pd.DataFrame(speed.T, columns=['train', 'valid'])
        df['t'] = range(len(df))
        df = df.melt(id_vars='t', value_vars=['train', 'valid'], var_name='split', value_name='distance')

        p = (ggplot(df, aes(x='t', y='distance', color='split')) 
        + geom_vline(xintercept=self.config.num_steps, linetype='dashed', color='black')
        + geom_line(size=1)
        + labs(x='Step', y='Speed')
        + theme_bw()
        )
        return p, df
