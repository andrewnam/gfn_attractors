import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import itertools
from functools import cached_property
import string

from .image_datamodule import ImageDataModule


class CharGridDataModule(ImageDataModule):

    FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

    def __init__(self, panel_size, strings=None, test_strings=None, characters=None, n=None, f_validation=.1, seed=0, batch_size=128):
        assert strings is not None or characters is not None
        super().__init__(seed, batch_size)

        self.panel_size = panel_size
        if strings is not None:
            self.characters = ''.join(sorted(set([c for s in strings for c in s])))
            if test_strings is None:
                self.strings, self.test_strings = self.generate_default_test_strings(strings)
            else:
                self.test_strings = test_strings
        else:
            self.characters = characters
            strings = self.generate_random_strings(characters, n)
            self.strings, self.test_strings = self.generate_default_test_strings(strings)
            
        self.f_validation = f_validation

    @classmethod
    def create_letter_transitions_data(cls, panel_size, n, p_stay=.1, p_forward=.6, p_backward=.3, seed=0, **kwargs):
        rng = np.random.default_rng(seed)
        characters = string.ascii_uppercase

        probs = [p_backward, p_stay, p_forward]
        k = len(characters) - 1
        for i in range(k):
            probs.append(probs[-1] * p_forward)
            probs.insert(0, probs[0] * p_backward)
        probs = np.array(probs)
        probs /= probs.sum()

        indices = rng.integers(0, len(characters), size=n)
        offsets = rng.choice(np.arange(0, len(probs)) - len(probs)//2, 3*n, p=probs).reshape((n, 3))
        indices = np.concatenate([indices.reshape((n, 1)), offsets], axis=1)
        indices = np.cumsum(indices, axis=1) % len(characters)
        strings = [''.join(characters[j] for j in i) for i in indices]
        data_module = cls(panel_size=panel_size, strings=strings, seed=seed, **kwargs)
        data_module._probs = probs
        return data_module


    @cached_property
    def char_images(self):
        char_imgs = {}
        size = self.panel_size
        font = ImageFont.truetype(self.FONT_PATH, int(size * 3 / 4))
        for c in ' ' + self.characters:
            img = Image.fromarray(np.zeros((size, size), dtype=bool))
            draw = ImageDraw.Draw(img)
            _, _, w, h = draw.textbbox((0, 0), c, font=font)
            draw.text(((size-w)/2, (size-h)/2), c, font=font, fill='white')
            char_imgs[c] = np.array(img)
        return char_imgs
    
    def generate_random_strings(self, characters, n=None):
        if n is None:
            strings = list(itertools.product(self.characters, repeat=4))
        else:
            strings = set()
            characters = list(self.characters)
            while len(strings) < n:
                strings.add(''.join(self.rng.choice(characters, size=4)))
        return list(strings)
    
    def generate_default_test_strings(self, strings):
        test_chars = self.characters[-(len(self.characters) // 4):]
        test_strings = [s for s in strings if s[0] in test_chars]
        strings = [s for s in strings if s[0] not in test_chars]
        return strings, test_strings
    
    def init_df_data(self, strings, test_strings):
        split = ['valid' if self.rng.random() < self.f_validation else 'train' for _ in strings]
        split += ['test'] * len(test_strings)
        strings = strings + test_strings
        df_data = pd.DataFrame({'uuid': range(len(strings)), 'split': split, 'string': strings})
        return df_data
    
    def create_image(self, string):
        images = [self.char_images[c] for c in string]
        im1 = np.concatenate(images[:2], axis=1)
        im2 = np.concatenate(images[2:], axis=1)
        im = np.concatenate([im1, im2], axis=0)
        return im
    
    def prepare_data(self):
        self.df_data = self.init_df_data(self.strings, self.test_strings)
        images = np.stack([self.create_image(string) for string in self.df_data.string])
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        self.train_indices = torch.tensor(self.df_data[self.df_data.split == 'train'].uuid.values)
        self.valid_indices = torch.tensor(self.df_data[self.df_data.split == 'valid'].uuid.values)
        self.test_indices = torch.tensor(self.df_data[self.df_data.split == 'test'].uuid.values)
