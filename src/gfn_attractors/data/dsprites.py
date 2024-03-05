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

    def __init__(self, 
                 size=64,
                 colorize=True,
                 constant_orientation=True,
                 min_scale=0,
                 pos_stride=1,
                 f_validation=.1,
                 num_pos_tokens: int = None,
                 label_sentence_pad_tokens=True,
                 label_sentence_eos_tokens=True,
                 path=None,
                 **kwargs):
        """
        num_pos_tokens: if not None, label sentences are included in the dataloaders
        """
        super().__init__(**kwargs)
        self._size = size
        self.colorize = colorize
        self.constant_orientation = constant_orientation
        self.min_scale = min_scale
        self.pos_stride = pos_stride
        self.f_validation = f_validation
        self.num_pos_tokens = num_pos_tokens
        self.label_sentence_pad_tokens = label_sentence_pad_tokens
        self.label_sentence_eos_tokens = label_sentence_eos_tokens
        self.path = Path(path).expanduser() if path is not None else None
        
        self.label_sentences = None

    @property
    def df_data(self):
        return self.df_labels

    @classmethod
    def load_raw_data(cls, constant_orientation=True, min_scale=0, pos_stride=1):
        with np.load(cls.RAW_DATA_PATH) as raw_data:
            images = raw_data['imgs']
            labels = raw_data['latents_classes']
            latents = raw_data['latents_values']

        if constant_orientation:
            # Remove all the images with orientation != 0
            keep = labels[:, 3] == 0
        if min_scale > 0:
            keep &= labels[:, 2] >= min_scale
        # Only keep images with positions in 5x5 grid
        keep &= np.logical_or(labels[:,-1] % pos_stride == 0, labels[:,-1] == 31)
        keep &= np.logical_or(labels[:,-2] % pos_stride == 0, labels[:,-2] == 31)
        
        images = images[keep]
        labels = labels[keep]
        latents = latents[keep]

        # Remove color (they're all white) and remove orientation since we only kept the ones with orientation == 0
        labels = np.concatenate([labels[:, 1:3], labels[:, 4:]], axis=-1)
        latents = np.concatenate([latents[:, 1:3], latents[:, 4:]], axis=-1)
        return images, labels, latents
    
    def prepare_data(self):
        if self.path is not None:
            self.df_labels = pd.read_csv(self.path / 'labels.tsv', sep='\t')
            self.df_latents = pd.read_csv(self.path / 'latents.tsv', sep='\t')
            self.images = torch.load(self.path / 'images.pt')
            return 
        
        images, labels, latents = self.load_raw_data(self.constant_orientation, self.min_scale, self.pos_stride)
        if self._size != 64:
            images = transforms.Resize(self._size)(torch.tensor(images)).numpy()
        images, labels, latents = self.color_images(images, labels, latents)

        df_latents = pd.DataFrame(latents, columns=['color', 'obj_shape', 'scale', 'x', 'y']).reset_index(names='uuid')
        df_labels = pd.DataFrame(labels, columns=['color_id', 'obj_shape_id', 'scale', 'x', 'y']).reset_index(names='uuid')
        df_latents.color = df_latents.color.astype('int')
        df_latents.obj_shape = df_latents.obj_shape.astype('int')
        df_latents[['r', 'g', 'b']] = self.COLORS[df_latents.color]
        df_labels['color'] = self.COLOR_NAMES[df_labels.color_id]
        df_labels['obj_shape'] = df_labels.obj_shape_id.apply(lambda x: ['square', 'oval', 'heart'][x])
        df_labels[['r', 'g', 'b']] = self.COLORS[df_labels.color_id]

        self.df_latents = df_latents
        self.df_labels = df_labels

        self.images = torch.tensor(images).float()
        self.latents = torch.tensor(self.df_latents.values[:,1:]) # exclude the index
        self.labels = torch.tensor(self.df_labels.select_dtypes(include=np.number).values)
        if self.num_pos_tokens is not None:
            self.label_sentences = self.get_label_sentences(self.num_pos_tokens, 
                                                            pad_tokens=self.label_sentence_pad_tokens, 
                                                            eos_tokens=self.label_sentence_eos_tokens)

    def color_images(self, images, latents, labels):
        num_colors = len(self.COLORS) if self.colorize else 1
        images = einops.repeat(images, 'b h w -> b k c h w', c=3, k=num_colors)
        images = [images[:, i] * self.COLORS[i].reshape(1, 3, 1, 1) for i in range(num_colors)]
        images = np.concatenate(images, axis=0)

        color_labels = einops.repeat(np.arange(num_colors), 'k -> (k b) 1', b=len(latents))
        latents = einops.repeat(latents, 'b a -> (k b) a', k=num_colors)
        latents = np.concatenate([color_labels, latents], axis=1)
        labels = einops.repeat(labels, 'b a -> (k b) a', k=num_colors)
        labels = np.concatenate([color_labels, labels], axis=1)
        return images, latents, labels

    def setup(self, stage: str='fit') -> None:
        if stage == "fit":
            generator = torch.Generator().manual_seed(self.seed)
            train_indices, valid_indices = random_split(range(len(self.images)), [1-self.f_validation, self.f_validation], generator=generator)
            self.train_indices = torch.tensor(train_indices.indices)
            self.valid_indices = torch.tensor(valid_indices.indices)

    def get_dataloader_item(self, index):
        item = {'image': self.images[index],
                'latent': self.latents[index],
                'label': self.labels[index],
                'index': index}
        if self.label_sentences is not None:
            item['label_sentence'] = self.label_sentences[index]
        return item
    
    def get_label_sentences(self, num_pos_tokens, pad_tokens=True, eos_tokens=True):
        labels = self.labels.clone()[:,1:-3] # remove obj_id and RGB
        labels[:,3] = ((num_pos_tokens - 1) * labels[:,3] / labels[:,3].max()).round() # x-position
        labels[:,4] = ((num_pos_tokens - 1) * labels[:,4] / labels[:,4].max()).round() # y-position
        for i in range(1, labels.shape[1]):
            labels[:,i] += 1 + labels[:,i-1].max()
        if eos_tokens:
            labels += 1
            labels = F.pad(labels, (0, 1))
        if pad_tokens:
            labels += 1
        return labels
    
    def get_feature_labels(self):
        feature_names = ['color_id', 'obj_shape_id', 'scale', 'x', 'y', 'r', 'g', 'b']
        num_feature_labels = [len(self.df_labels[feature].unique()) for feature in feature_names]
        return feature_names, num_feature_labels

    def get_feature_vectors(self):
        features = []
        for column in ['color_id', 'obj_shape_id', 'scale', 'x', 'y', 'r', 'g', 'b']:
            values = sorted(self.df_labels[column].unique())
            value_indices = {v: i for i, v in enumerate(values)}
            values = [value_indices[v] for v in self.df_labels[column]]
            features.append(np.array(values))
        features = np.stack(features, axis=1)
        return torch.tensor(features, dtype=int)



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

        # Hold out all yellow objects in (0, 3) region
        # self.test_xy_mode_colors = torch.zeros((len(self), 7), dtype=bool)
        if self.holdout_xy_mode_color:
            xy_mode_keep = self.labels[:,2] < 8
            xy_mode_keep &= self.labels[:,3] >= 24
            self.test_xy_mode_color_indices = torch.arange(len(self))[xy_mode_keep]
            self.train_colors[xy_mode_keep, 3] = 0
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
        
        
    def create_batch(self, indices, color_set='train', rgb: tuple = None, color_noise=True):
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

        return {'index': torch.tensor(indices), 
                'image': torch.tensor(batch_images).float(), 
                'label': torch.tensor(batch_labels).float(),
                'latent': torch.tensor(batch_latents).float()}
    
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
        return {k: v[0] for k, v in self.__getitems__([index]).items()}

    def __getitems__(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return self.create_batch(indices)

# class ContinuousDSpritesDataModule(ImageDataModule):

#     RAW_DATA_PATH = os.path.join(git.Repo('.', search_parent_directories=True).working_tree_dir, 
#                                  'data/raw/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

#     COLORS = np.array([
#         (1, 1, 1), # white
#         (1, 0, 0), # red
#         (0, 1, 0), # green
#         (0, 0, 1), # blue
#         (1, 1, 0), # yellow
#         (1, 0, 1), # magenta
#         (0, 1, 1), # cyan
#     ])

#     COLOR_NAMES = np.array([
#         'white',
#         'red',
#         'green',
#         'blue',
#         'yellow',
#         'magenta',
#         'cyan',
#     ])

#     LABEL_SENTENCE_FEATURES = ['color', 'obj_shape', 'scale', 'x', 'y']
#     POSITION_WEIGHTS = np.array([2, 3, 4, 5, 4, 3, 2, 1, 
#                                  2, 3, 4, 5, 4, 3, 2, 1, 
#                                  2, 3, 4, 5, 4, 3, 2, 1, 
#                                  2, 3, 4, 5, 4, 3, 2])
#     rgb = np.array([[1, 0, 0],
#                     [0, 1, 0],
#                     [0, 0, 1],
#                     [1, 1, 0],
#                     [1, 0, 1],
#                     [0, 1, 1],
#                     [1, 1, 1]], dtype=float)

#     def __init__(self, 
#                  size=64,
#                  constant_orientation=True,
#                  min_scale=0,
#                  f_validation=.1,
#                  **kwargs):
#         """
#         num_pos_tokens: if not None, label sentences are included in the dataloaders
#         """
#         super().__init__(**kwargs)
#         self._size = size
#         self.constant_orientation = constant_orientation
#         self.min_scale = min_scale
#         self.f_validation = f_validation

#     @property
#     def num_channels(self):
#         return 3

#     @classmethod
#     def load_raw_data(cls, constant_orientation=True, min_scale=0, size=64):
#         with np.load(cls.RAW_DATA_PATH) as raw_data:
#             images = raw_data['imgs']
#             labels = raw_data['latents_classes']
#             latents = raw_data['latents_values']

#         if constant_orientation:
#             keep = labels[:, 3] == 0
#         if min_scale > 0:
#             keep &= labels[:, 2] >= min_scale
        
#         keep &= labels[:,-1] < 31
#         keep &= labels[:,-2] < 31
#         images = images[keep]
#         labels = labels[keep][:, [1, 2, 4, 5]]
#         latents = latents[keep][:, [1, 2, 4, 5]]

#         if size != 64:
#             images = transforms.Resize(size)(torch.tensor(images)).numpy()
#         return images, labels, latents
    
#     def prepare_data(self):
#         self.images, self.labels, self.latents = self.load_raw_data(self.constant_orientation, self.min_scale, self._size)
#         generator = torch.Generator().manual_seed(self.seed)
#         train_indices, valid_indices = random_split(range(len(self.images)), [1-self.f_validation, self.f_validation], generator=generator)
#         self.train_indices = torch.tensor(train_indices.indices)
#         self.valid_indices = torch.tensor(valid_indices.indices)
#         sample_weights = self.POSITION_WEIGHTS[self.labels[:,-2]] * self.POSITION_WEIGHTS[self.labels[:,-1]]
#         sample_weights[valid_indices.indices] = 0
#         self.sample_weights = sample_weights / sample_weights.sum()
        
#     def create_batch(self, indices):
#         # batch_size = len(indices)
#         batch_images = self.images[indices]
#         batch_images = einops.repeat(batch_images, 'b x y -> b 3 x y')
#         batch_latents = self.latents[indices]

#         colors = self.rgb[np.random.randint(0, len(self.rgb), len(batch_images))]
#         noise = torch.distributions.HalfNormal(.2).sample((len(batch_images), 3)).numpy()
#         colors += noise * (-2*colors + 1)
#         batch_latents = np.concatenate([colors, batch_latents], axis=1)
#         colors = einops.repeat(colors, 'b c -> b c 1 1')
#         batch_images = batch_images * colors
#         return torch.tensor(batch_images).float(), torch.tensor(batch_latents).float()
    
#     def train_dataloader(self, batch_size=None) -> DataLoader:
#         sampler = WeightedRandomSampler(self.sample_weights, len(self.sample_weights))
#         batch_size = self.batch_size if batch_size is None else batch_size
#         return DataLoader(self, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: x, 
#                           drop_last=True)
    
#     def valid_dataloader(self, batch_size=None) -> DataLoader:
#         sampler = RandomSampler2(self.valid_indices, replacement=False)
#         batch_size = self.batch_size if batch_size is None else batch_size
#         return DataLoader(self, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: x, drop_last=True)
    
#     def __getitem__(self, index):
#         return {k: v[0] for k, v in self.__getitems__([index]).items()}

#     def __getitems__(self, indices):
#         if isinstance(indices, torch.Tensor):
#             indices = indices.tolist()
#         batch_images, batch_latents = self.create_batch(indices)
#         return {'index': torch.tensor(indices), 'image': batch_images, 'latent': batch_latents}