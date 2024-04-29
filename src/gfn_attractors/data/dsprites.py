import os
import pandas as pd
import numpy as np
from pathlib import Path
import git

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split
import einops
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchdata.datapipes.map import SequenceWrapper
from ..misc.torch_utils import RandomSampler2

from .image_datamodule import ImageDataModule


class DSpritesDataModule(ImageDataModule):

    RAW_DATA_PATH = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir, 
                                 'data/raw/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    COLORS = np.array([
        (1, 1, 1), # white
        (1, 0, 0), # red
        (0, 1, 0), # green
        (0, 0, 1), # blue
        (1, 1, 0), # yellow
        (1, 0, 1), # magenta
        (0, 1, 1), # cyan
    ])

    COLOR_NAMES = np.array([
        'white',
        'red',
        'green',
        'blue',
        'yellow',
        'magenta',
        'cyan',
    ])

    LABEL_SENTENCE_FEATURES = ['color', 'obj_shape', 'scale', 'x', 'y']
    POSITION_WEIGHTS = np.array([2, 3, 4, 5, 4, 3, 2, 1, 
                                 2, 3, 4, 5, 4, 3, 2, 1, 
                                 2, 3, 4, 5, 4, 3, 2, 1, 
                                 2, 3, 4, 5, 4, 3, 2])
    rgb = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1]], dtype=float)

    def __init__(self, 
                 size=64,
                 constant_orientation=True,
                 min_scale=0,
                 f_validation=.1,
                 holdout_xy_mode=False,
                 holdout_xy_nonmode=False,
                 holdout_xy_shape=False,
                 holdout_xy_mode_color=False,
                 holdout_shape_color=False,
                 **kwargs):
        """
        num_pos_tokens: if not None, label sentences are included in the dataloaders
        holdout_xy_mode: if True, hold out all data in (1, 2) region
        holdout_xy_nonmode: if True, hold out all data from a single position in (2, 1) region
        holdout_xy_shape: if True, hold out all squares in (3, 0) region
        holdout_xy_mode_color: if True, hold out all yellow objects in (0, 3) region
        holdout_shape_color: if True, holdout all magenta ovals
        """
        super().__init__(**kwargs)
        self._size = size
        self.constant_orientation = constant_orientation
        self.min_scale = min_scale
        self.f_validation = f_validation
        self.holdout_xy_mode = holdout_xy_mode
        self.holdout_xy_nonmode = holdout_xy_nonmode
        self.holdout_xy_shape = holdout_xy_shape
        self.holdout_xy_mode_color = holdout_xy_mode_color
        self.holdout_shape_color = holdout_shape_color

    @property
    def num_channels(self):
        return 3

    @classmethod
    def load_raw_data(cls, constant_orientation=True, min_scale=0, size=64):
        with np.load(cls.RAW_DATA_PATH) as raw_data:
            images = raw_data['imgs']
            labels = raw_data['latents_classes']
            latents = raw_data['latents_values']

        if constant_orientation:
            keep = labels[:, 3] == 0
        if min_scale > 0:
            keep &= labels[:, 2] >= min_scale
        
        keep &= labels[:,-1] < 31
        keep &= labels[:,-2] < 31
        images = images[keep]
        labels = labels[keep][:, [1, 2, 4, 5]]
        latents = latents[keep][:, [1, 2, 4, 5]]
        latents[:,1] = (latents[:,1] - .5) / .5 # rescale from [0.5, 1.0] to [0.0, 1.0]

        if size != 64:
            images = transforms.Resize(size)(torch.tensor(images)).numpy()
        return images, labels, latents
    
    def prepare_data(self):
        images, labels, latents = self.load_raw_data(self.constant_orientation, self.min_scale, self._size)
        images = self.rgb.reshape(1, 7, 3, 1, 1) * images.reshape(-1, 1, 1, self._size, self._size)
        images = images.reshape(-1, 3, self._size, self._size)

        rgb = einops.repeat(self.rgb, 'k c -> (b k) c', b=len(labels))
        labels = einops.repeat(labels, 'b l -> (b k) l', k=len(self.rgb))
        labels = np.concatenate([rgb, labels], axis=-1).astype(int)
        latents = einops.repeat(latents, 'b l -> (b k) l', k=len(self.rgb))
        latents = np.concatenate([rgb, latents], axis=-1)
        
        self.images = images
        self.labels = labels
        self.latents = latents

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.split_dataset()
        sample_weights = self.POSITION_WEIGHTS[self.labels[:,-2]] * self.POSITION_WEIGHTS[self.labels[:,-1]]
        sample_weights[self.valid_indices] = 0
        self.sample_weights = sample_weights / sample_weights.sum()

    def split_dataset(self):
        self.train_colors = torch.ones((len(self), 7), dtype=bool)
        self.test_colors = torch.ones((len(self), 7), dtype=bool)

        # Hold out all data in (1, 2) region
        xy_mode_keep = self.labels[:,-2] >= 8
        xy_mode_keep &= self.labels[:,-2] < 16
        xy_mode_keep &= self.labels[:,-1] >= 16
        xy_mode_keep &= self.labels[:,-1] < 24
        self.test_xy_mode_indices = torch.arange(len(self))[xy_mode_keep]

        # Hold out all data from a single position in (2, 1) region
        xy_nonmode_keep = self.labels[:,-2] == 21 
        xy_nonmode_keep &= self.labels[:,-1] == 9
        self.test_xy_nonmode_indices = torch.arange(len(self))[xy_nonmode_keep]

        # Hold out all squares in (3, 0) region
        xy_shape_keep = self.labels[:,3] == 0
        xy_shape_keep &= self.labels[:,-2] >= 24
        xy_shape_keep &= self.labels[:,-1] < 8
        self.test_xy_shape_indices = torch.arange(len(self))[xy_shape_keep]

        # Hold out all green objects in (0, 3) region
        # self.test_xy_mode_colors = torch.zeros((len(self), 7), dtype=bool)
        if self.holdout_xy_mode_color:
            xy_mode_keep = self.labels[:,-2] < 8
            xy_mode_keep &= self.labels[:,-1] >= 24
            xy_mode_keep &= self.labels[:,0] == 1
            xy_mode_keep &= self.labels[:,1] == 1
            xy_mode_keep &= self.labels[:,2] == 0
            self.test_xy_mode_color_indices = torch.arange(len(self))[xy_mode_keep]

        # Holdout all magenta ovals
        if self.holdout_shape_color:
            shape_color_keep = self.labels[:,3] == 1
            shape_color_keep &= self.labels[:,0] == 1
            shape_color_keep &= self.labels[:,1] == 0
            shape_color_keep &= self.labels[:,2] == 1
            self.test_shape_color_indices = torch.arange(len(self))[shape_color_keep]
            
        # Train and valid indices
        train_keep = torch.ones(len(self), dtype=bool)
        if self.holdout_xy_mode:
            train_keep[self.test_xy_mode_indices] = 0
        if self.holdout_xy_nonmode:
            train_keep[self.test_xy_nonmode_indices] = 0
        if self.holdout_xy_shape:
            train_keep[self.test_xy_shape_indices] = 0
        if self.holdout_xy_mode_color:
            train_keep[self.test_xy_mode_color_indices] = 0
        if self.holdout_shape_color:
            train_keep[self.test_shape_color_indices] = 0

        self.test_indices = torch.arange(len(self))[~train_keep]
        nontest_indices = torch.arange(len(self))[train_keep]
        generator = torch.Generator().manual_seed(self.seed)
        train_indices, valid_indices = random_split(range(len(nontest_indices)), [1-self.f_validation, self.f_validation], generator=generator)
        self.train_indices = nontest_indices[train_indices.indices]
        self.valid_indices = nontest_indices[valid_indices.indices]
        
    def create_batch(self, indices, color_noise=True, device='cpu'):
        """
        color_set: either 'train', 'test', or 'both'
        rgb: tuple of three 1s or 0s, indicating whether red, green, and blue are included
        """
        batch_images = self.images[indices]
        batch_latents = self.latents[indices]
        batch_labels = self.labels[indices]
        color_modes = batch_labels[:,:3]

        noise = torch.distributions.HalfNormal(.2).sample((len(batch_images), 3)).numpy()
        colors = color_modes + color_noise * noise * (-2*color_modes + 1)
        batch_latents[:,:3] = colors
        
        features = np.concatenate([batch_labels[:,:4], batch_latents[:,4:]], axis=1)
        return {'index': torch.tensor(indices, device=device), 
                'x': torch.tensor(batch_images, device=device).float(), 
                'label': torch.tensor(batch_labels, device=device).float(),
                'latent': torch.tensor(batch_latents, device=device).float(),
                'features': torch.tensor(features, device=device).float()}
    
    def train_dataloader(self, batch_size=None) -> DataLoader:
        sampler = WeightedRandomSampler(self.sample_weights, len(self.sample_weights))
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: x, 
                          drop_last=True)
    
    def valid_dataloader(self, batch_size=None) -> DataLoader:
        sampler = RandomSampler2(self.valid_indices, replacement=False)
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: x, drop_last=True)

    
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
    



class ContinuousDSpritesDataModule(ImageDataModule):

    RAW_DATA_PATH = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir, 
                                 'data/raw/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    COLORS = np.array([
        (1, 1, 1), # white
        (1, 0, 0), # red
        (0, 1, 0), # green
        (0, 0, 1), # blue
        (1, 1, 0), # yellow
        (1, 0, 1), # magenta
        (0, 1, 1), # cyan
    ])

    COLOR_NAMES = np.array([
        'white',
        'red',
        'green',
        'blue',
        'yellow',
        'magenta',
        'cyan',
    ])

    LABEL_SENTENCE_FEATURES = ['color', 'obj_shape', 'scale', 'x', 'y']
    POSITION_WEIGHTS = np.array([2, 3, 4, 5, 4, 3, 2, 1, 
                                 2, 3, 4, 5, 4, 3, 2, 1, 
                                 2, 3, 4, 5, 4, 3, 2, 1, 
                                 2, 3, 4, 5, 4, 3, 2])
    rgb = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1]], dtype=float)

    def __init__(self, 
                 size=64,
                 constant_orientation=True,
                 min_scale=0,
                 f_validation=.1,
                 holdout_xy_mode=False,
                 holdout_xy_nonmode=False,
                 holdout_xy_shape=False,
                 holdout_xy_mode_color=False,
                 holdout_shape_color=False,
                 **kwargs):
        """
        num_pos_tokens: if not None, label sentences are included in the dataloaders
        holdout_xy_mode: if True, hold out all data in (1, 2) region
        holdout_xy_nonmode: if True, hold out all data from a single position in (2, 1) region
        holdout_xy_shape: if True, hold out all squares in (3, 0) region
        holdout_xy_mode_color: if True, hold out all yellow objects in (0, 3) region
        holdout_shape_color: if True, holdout all magenta ovals
        """
        super().__init__(**kwargs)
        self._size = size
        self.constant_orientation = constant_orientation
        self.min_scale = min_scale
        self.f_validation = f_validation
        self.holdout_xy_mode = holdout_xy_mode
        self.holdout_xy_nonmode = holdout_xy_nonmode
        self.holdout_xy_shape = holdout_xy_shape
        self.holdout_xy_mode_color = holdout_xy_mode_color
        self.holdout_shape_color = holdout_shape_color

    @property
    def num_channels(self):
        return 3

    @classmethod
    def load_raw_data(cls, constant_orientation=True, min_scale=0, size=64):
        with np.load(cls.RAW_DATA_PATH) as raw_data:
            images = raw_data['imgs']
            labels = raw_data['latents_classes']
            latents = raw_data['latents_values']

        if constant_orientation:
            keep = labels[:, 3] == 0
        if min_scale > 0:
            keep &= labels[:, 2] >= min_scale
        
        keep &= labels[:,-1] < 31
        keep &= labels[:,-2] < 31
        images = images[keep]
        labels = labels[keep][:, [1, 2, 4, 5]]
        latents = latents[keep][:, [1, 2, 4, 5]]
        latents[:,1] = (latents[:,1] - .5) / .5 # rescale from [0.5, 1.0] to [0.0, 1.0]

        if size != 64:
            images = transforms.Resize(size)(torch.tensor(images)).numpy()
        return images, labels, latents
    
    def prepare_data(self):
        self.images, self.labels, self.latents = self.load_raw_data(self.constant_orientation, self.min_scale, self._size)
        self.split_dataset()
        sample_weights = self.POSITION_WEIGHTS[self.labels[:,-2]] * self.POSITION_WEIGHTS[self.labels[:,-1]]
        sample_weights[self.valid_indices] = 0
        self.sample_weights = sample_weights / sample_weights.sum()

    def split_dataset(self):
        self.train_colors = torch.ones((len(self), 7), dtype=bool)
        self.test_colors = torch.ones((len(self), 7), dtype=bool)

        # Hold out all data in (1, 2) region
        xy_mode_keep = self.labels[:,2] >= 8
        xy_mode_keep &= self.labels[:,2] < 16
        xy_mode_keep &= self.labels[:,3] >= 16
        xy_mode_keep &= self.labels[:,3] < 24
        self.test_xy_mode_indices = torch.arange(len(self))[xy_mode_keep]

        # Hold out all data from a single position in (2, 1) region
        xy_nonmode_keep = self.labels[:,2] == 21 
        xy_nonmode_keep &= self.labels[:,3] == 9
        self.test_xy_nonmode_indices = torch.arange(len(self))[xy_nonmode_keep]

        # Hold out all squares in (3, 0) region
        xy_shape_keep = self.labels[:,0] == 0
        xy_shape_keep &= self.labels[:,2] >= 24
        xy_shape_keep &= self.labels[:,3] < 8
        self.test_xy_shape_indices = torch.arange(len(self))[xy_shape_keep]

        # Hold out all green objects in (0, 3) region
        # self.test_xy_mode_colors = torch.zeros((len(self), 7), dtype=bool)
        if self.holdout_xy_mode_color:
            xy_mode_keep = self.labels[:,2] < 8
            xy_mode_keep &= self.labels[:,3] >= 24
            self.test_xy_mode_color_indices = torch.arange(len(self))[xy_mode_keep]
            self.train_colors[xy_mode_keep, 1] = 0
            self.test_colors[xy_mode_keep] = 0

        # Holdout all magenta ovals
        if self.holdout_shape_color:
            shape_color_keep = self.labels[:,0] == 1
            self.test_shape_color_indices = torch.arange(len(self))[shape_color_keep]
            self.train_colors[shape_color_keep, 4] = 0
            self.test_colors[shape_color_keep] = 0

        if self.holdout_xy_mode_color:
            self.test_colors[xy_mode_keep, 3] = 1
        if self.holdout_shape_color:
            self.test_colors[shape_color_keep, 4] = 1

        # Train and valid indices
        train_keep = torch.ones(len(self), dtype=bool)
        if self.holdout_xy_mode:
            train_keep[self.test_xy_mode_indices] = 0
        if self.holdout_xy_nonmode:
            train_keep[self.test_xy_nonmode_indices] = 0
        if self.holdout_xy_shape:
            train_keep[self.test_xy_shape_indices] = 0

        self.test_indices = torch.arange(len(self))[~train_keep]
        nontest_indices = torch.arange(len(self))[train_keep]
        generator = torch.Generator().manual_seed(self.seed)
        train_indices, valid_indices = random_split(range(len(nontest_indices)), [1-self.f_validation, self.f_validation], generator=generator)
        self.train_indices = torch.tensor(nontest_indices[train_indices.indices])
        self.valid_indices = torch.tensor(nontest_indices[valid_indices.indices])
        
    def create_batch(self, indices, color_set='train', rgb: tuple = None, color_noise=True, scale_group_size=2, position_group_size=8, device='cpu'):
        """
        color_set: either 'train', 'test', or 'both'
        rgb: tuple of three 1s or 0s, indicating whether red, green, and blue are included
        """
        batch_images = self.images[indices]
        batch_images = einops.repeat(batch_images, 'b x y -> b 3 x y')
        batch_latents = self.latents[indices]
        batch_labels = self.labels[indices]

        if rgb is not None:
            color_modes = np.array(rgb, dtype=float)
            color_modes = einops.repeat(color_modes, 'c -> b c', b=len(batch_images))
        else:
            if color_set == 'both':
                color_indices = torch.randint(0, len(self.rgb), (len(batch_images),))
            else:
                color_indices = self.test_colors[indices] if color_set == 'test' else self.train_colors[indices]
                color_indices = torch.distributions.Categorical(color_indices.float()).sample()
            color_modes = self.rgb[color_indices]
            if color_modes.ndim == 1:
                color_modes = np.expand_dims(color_modes, 0)
        noise = torch.distributions.HalfNormal(.2).sample((len(batch_images), 3)).numpy()
        colors = color_modes + color_noise * noise * (-2*color_modes + 1)
        # print(color_modes.shape, batch_labels.shape)
        # print(indices)
        batch_latents = np.concatenate([colors, batch_latents], axis=1)
        batch_labels = np.concatenate([color_modes, batch_labels], axis=1)
        colors = einops.repeat(colors, 'b c -> b c 1 1')
        batch_images = batch_images * colors


        features = np.array(batch_labels)
        features[:,4] = features[:,4] // 2
        features[:,5] = features[:,5] // 8
        features[:,6] = features[:,6] // 8

        return {'index': torch.tensor(indices, device=device), 
                'x': torch.tensor(batch_images, device=device).float(), 
                'label': torch.tensor(batch_labels, device=device).float(),
                'latent': torch.tensor(batch_latents, device=device).float(),
                'features': torch.tensor(features, device=device).float()}
    
    def train_dataloader(self, batch_size=None) -> DataLoader:
        sampler = WeightedRandomSampler(self.sample_weights, len(self.sample_weights))
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: x, 
                          drop_last=True)
    
    def valid_dataloader(self, batch_size=None) -> DataLoader:
        sampler = RandomSampler2(self.valid_indices, replacement=False)
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(self, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: x, drop_last=True)

    
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
    