import numpy as np
from pathlib import Path
import git

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.map import SequenceWrapper
from PIL import Image, ImageDraw, ImageFont


class ImageDataModule(pl.LightningDataModule):
    """
    Base class for image datasets.
    During prepare_data(), should define self.images and self.train_indices.
    """

    FONT_PATH = Path(git.Repo('.', search_parent_directories=True).working_tree_dir) / 'data/fonts/'

    def __init__(self,
                 seed=0,
                 batch_size=128):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

        self.images = None
        self.train_indices = None
        self.valid_indices = None
        self.test_indices = None

    @property
    def size(self):
        return self.images.shape[-1]
    
    @property
    def num_channels(self):
        return self.images.shape[1]

    def setup(self, stage=None):
        pass

    def get_dataloader_item(self, index):
        item = {'image': self.images[index],
                'index': index}
        return item
    
    def full_dataloader(self, batch_size=None) -> DataLoader:
        dp = SequenceWrapper(torch.arange(len(self.images)))
        dp = dp.map(lambda i: self.get_dataloader_item(i))
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(dp, batch_size=batch_size, shuffle=True)
        
    def train_dataloader(self, batch_size=None) -> DataLoader:
        dp = SequenceWrapper(self.train_indices)
        dp = dp.map(lambda i: self.get_dataloader_item(i))
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(dp, batch_size=batch_size, shuffle=True)
    
    def val_dataloader(self, batch_size=None):
        if self.valid_indices is not None:
            dp = SequenceWrapper(self.valid_indices)
            dp = dp.map(lambda i: self.get_dataloader_item(i))
            batch_size = self.batch_size if batch_size is None else batch_size
            return DataLoader(dp, batch_size=batch_size, shuffle=True)
        raise Exception('No validation set')
    
    def test_dataloader(self, batch_size=None):
        if self.test_indices is not None:
            dp = SequenceWrapper(self.test_indices)
            dp = dp.map(lambda i: self.get_dataloader_item(i))
            batch_size = self.batch_size if batch_size is None else batch_size
            return DataLoader(dp, batch_size=batch_size, shuffle=True)
        raise Exception('No test set')

    def render(self, img: int | torch.Tensor, size=None,
               text=None, text_size=24, text_position=(4, 3), text_color=(255, 0, 0), 
               font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        """
        img: index of the image to render, or the image itself
            if img is a tensor, it should be of shape (size, size) or (1, size, size)
        """
        if isinstance(img, int) or len(img.shape) == 0:
            img = self.images[img]
        if img.shape[0] == 1:
            img = Image.fromarray(255 * img.squeeze(0).cpu().numpy()).convert('RGB')
        else:
            img = (255 * img).permute(1, 2, 0).cpu().numpy().round().astype('uint8')
            img = Image.fromarray(img)
        if size is not None:
            img = img.resize((size, size))

        if text is not None:
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, text_size)
            draw.text(text_position, text, font=font, fill=text_color)
        return img
    
    def __len__(self):
        return len(self.images)
