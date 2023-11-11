import numpy as np
import pandas as pd
from pathlib import Path
import git

import torch
from torchvision import transforms

from.image_datamodule import ImageDataModule


class Omniglot(ImageDataModule):

    def __init__(self,
                 size=None,
                 f_validation=.1,
                 num_test_alphabets=5,
                 num_crop_pixels=0,
                 path=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._size = size
        self.f_validation = f_validation
        self.num_test_alphabets = num_test_alphabets
        self.num_crop_pixels = num_crop_pixels
        
        if path is None:
            repo = git.Repo('.', search_parent_directories=True)
            repo_root = Path(repo.common_dir).parent
            self.path = repo_root / 'data/omniglot'
        else:
            self.path = Path(path).expanduser().resolve()

    def prepare_data(self):
        images = np.load(self.path / 'omniglot.npy')
        images = self.crop_edges(images, self.num_crop_pixels)
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        self.df_metadata = pd.read_csv(self.path / 'omniglot.tsv', sep='\t')

        if self._size is not None:
            self.images = transforms.Resize(self._size, antialias=True)(self.images)

        valid_id = set(self.df_metadata.groupby('alphabet').sample(frac=self.f_validation, random_state=self.rng).uuid)
        test_alphabets = self.rng.choice(self.df_metadata.alphabet.unique(), self.num_test_alphabets, replace=False)
        self.df_metadata['dataset'] = ['valid' if uuid in valid_id else 'test' if alphabet in test_alphabets else 'train' 
                                       for uuid, alphabet 
                                       in zip(self.df_metadata.uuid, self.df_metadata.alphabet)]

        self.train_indices = self.df_metadata[self.df_metadata.dataset == 'train'].uuid.values
        self.valid_indices = self.df_metadata[self.df_metadata.dataset == 'valid'].uuid.values
        self.test_indices = self.df_metadata[self.df_metadata.dataset == 'test'].uuid.values

    def crop_edges(self, images, num_pixels):
        """
        Compares the top and bottom rows, and the left and right columns,
        and removes the row/column with the most white pixels.
        """
        cropped = []
        for image in images:
            for i in range(num_pixels):
                image = image[1:] if image[0].sum() > image[-1].sum() else image[:-1]
                image = image[:,1:] if image[:,0].sum() > image[:,-1].sum() else image[:,:-1]
            cropped.append(image)
        return np.array(cropped)
