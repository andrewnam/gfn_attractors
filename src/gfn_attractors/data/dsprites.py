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

from .image_datamodule import ImageDataModule


class DSprites(ImageDataModule):

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
        df_latents[['r', 'g', 'b']] = DSprites.COLORS[df_latents.color]
        df_labels['color'] = DSprites.COLOR_NAMES[df_labels.color_id]
        df_labels['obj_shape'] = df_labels.obj_shape_id.apply(lambda x: ['square', 'oval', 'heart'][x])
        df_labels[['r', 'g', 'b']] = DSprites.COLORS[df_labels.color_id]

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
        num_colors = len(DSprites.COLORS) if self.colorize else 1
        images = einops.repeat(images, 'b h w -> b k c h w', c=3, k=num_colors)
        images = [images[:, i] * DSprites.COLORS[i].reshape(1, 3, 1, 1) for i in range(num_colors)]
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
